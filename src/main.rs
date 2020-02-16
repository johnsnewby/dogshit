#[macro_use]
extern crate lazy_static;

use actix_web::{get, App, Error, HttpRequest, HttpResponse, HttpServer, Responder};
use crossbeam_channel::bounded;
use futures::future::{ready, Ready};
use image::*;
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use rusttype::{FontCollection, Scale};
use std::io::{BufRead, Cursor, Read};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::{thread, time};
use tflite::ops::builtin::BuiltinOpResolver;
use tflite::{FlatBufferModel, Interpreter, InterpreterBuilder};

lazy_static! {
    static ref IMAGE: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec!()));
}

fn take_picture() -> std::io::Result<Vec<u8>> {
    let mut output = Command::new("raspistill")
        .args(&["-o", "-"])
        .stdout(Stdio::piped())
        .output()?;
    Ok(output.stdout)
}

#[actix_rt::main]
async fn main() -> std::io::Result<()> {
    let (sender, receiver) = bounded(1);

    if let Ok(mut img) = std::fs::File::open("test.jpg") {
        println!("Starting with test image");
        IMAGE.lock().unwrap().clear();
        img.read_to_end(&mut IMAGE.lock().unwrap()).unwrap();
        sender.try_send("foo").unwrap();
    }

    let labelmap: Vec<String> =
        std::io::BufReader::new(std::fs::File::open("labelmap.txt").unwrap())
            .lines()
            .map(|x| x.unwrap())
            .collect();
    println!("Labelmap: {:?}", labelmap);
    std::thread::spawn(move || loop {
        let output = take_picture().unwrap();
        IMAGE.lock().unwrap().clear();
        IMAGE.lock().unwrap().extend_from_slice(&output);
        match sender.try_send("foo") {
            Ok(_) => println!("Receiver was ready"),
            _ => println!("Receiver was not ready"),
        };
    });

    std::thread::spawn(move || find_objects(receiver, labelmap));

    HttpServer::new(|| App::new().service(latest))
        .bind("0.0.0.0:8088")?
        .run()
        .await
}

fn find_objects(receiver: crossbeam_channel::Receiver<&str>, labelmap: Vec<String>) {
    let model = FlatBufferModel::build_from_file("detect.tflite").unwrap();
    let resolver = BuiltinOpResolver::default();
    loop {
        println!("Waiting");
        receiver.recv().unwrap();
        println!("Received");
        let data = &*IMAGE.lock().unwrap();
        let img = image::load_from_memory(data.clone().as_slice()).unwrap();
        drop(data);
        let builder = InterpreterBuilder::new(&model, &resolver).unwrap();
        let mut interpreter = builder.build().unwrap();
        interpreter.allocate_tensors().unwrap();
        let inputs = interpreter.inputs();
        let inputs = inputs.clone();
        let outputs = interpreter.outputs();
        let outputs = outputs.clone();
        let boxes = outputs[0];
        let classes = outputs[1];
        let scores = outputs[2];
        let count = outputs[3];
        println!("inputs {:?} outputs {:?}", inputs, outputs);
        let input_info = interpreter.tensor_info(inputs[0]).unwrap();
        for i in 0..4 {
            let output_info = interpreter.tensor_info(outputs[i]).unwrap();
            println!("output_info {}: {:?}", i, input_info);
        }
        let resized = img.resize_to_fill(
            input_info.dims[1] as u32,
            input_info.dims[2] as u32,
            image::imageops::FilterType::Nearest,
        );
        resized.save("out.jpg").unwrap();
        let image_bytes = resized.as_rgb8().unwrap().to_vec();
        println!("image_bytes len = {}", image_bytes.len());
        let mut image_cursor = Cursor::new(image_bytes);
        image_cursor
            .read_exact(interpreter.tensor_data_mut(inputs[0]).unwrap())
            .unwrap();
        println!("Invoking");
        interpreter.invoke().unwrap();
        let counts: &[f32] = interpreter.tensor_data(count).unwrap();
        let boxes: &[f32] = interpreter.tensor_data(boxes).unwrap();
        println!("Boxes: {:?}", boxes);
        let classes: &[f32] = interpreter.tensor_data(classes).unwrap();
        let scores: &[f32] = interpreter.tensor_data(scores).unwrap();
        for i in 0..counts[0] as usize {
            println!(
                "result {} score {}, class {}",
                i, scores[i], labelmap[classes[i] as usize]
            );
        }
        let boxed = draw_boxes(
            0.4,
            img,
            counts[0] as usize,
            boxes,
            classes,
            scores,
            &labelmap,
        );
        boxed.save("boxed.jpg").unwrap();
    }
}

fn draw_boxes(
    threshold: f32,
    img: DynamicImage,
    count: usize,
    boxes: &[f32],
    classes: &[f32],
    scores: &[f32],
    labelmap: &Vec<String>,
) -> DynamicImage {
    let mut img = img.clone();
    let (width, height) = img.dimensions();
    let font = Vec::from(include_bytes!("LiberationSans-Regular.ttf") as &[u8]);
    let font = FontCollection::from_bytes(font)
        .unwrap()
        .into_font()
        .unwrap();
    for i in 0..count {
        if scores[i] > threshold {
            let (ymin, xmin, ymax, xmax) = (
                (height as f32 * boxes[4 * i]) as u32,
                (height as f32 * boxes[4 * i + 1]) as u32,
                (width as f32 * boxes[4 * i + 2]) as u32,
                (width as f32 * boxes[4 * i + 3]) as u32,
            );
            let rect = Rect::at(xmin as i32, ymin as i32).of_size(xmax - xmin, ymax - ymin);
            draw_hollow_rect_mut(&mut img, rect, Rgba([255u8, 255u8, 255u8, 128u8]));
            draw_text_mut(
                &mut img,
                Rgba([255u8, 255u8, 255u8, 128u8]),
                xmin,
                ymin,
                Scale { x: 20.0, y: 10.0 },
                &font,
                &labelmap[classes[i] as usize],
            );
        }
    }

    img
}

struct Image {
    bytes: Vec<u8>,
}

impl Responder for Image {
    type Error = Error;
    type Future = Ready<Result<HttpResponse, Error>>;

    fn respond_to(self, _req: &HttpRequest) -> Self::Future {
        ready(Ok(HttpResponse::Ok()
            .content_type("image/jpeg")
            .body(self.bytes)))
    }
}

#[get("/latest")]
async fn latest() -> impl Responder {
    Image {
        bytes: IMAGE.lock().unwrap().clone(),
    }
}
