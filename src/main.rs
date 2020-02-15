#[macro_use]
extern crate lazy_static;

use failure::Fallible;
use rascam::SimpleCamera;
use std::fs::File;
use std::io::Write;
use std::{thread, time};
use tflite::ops::builtin::BuiltinOpResolver;
use tflite::{FlatBufferModel, Interpreter, InterpreterBuilder};

fn init() -> InterpreterBuilder<'static> {
    let model = FlatBufferModel::build_from_file("detect.tflite").unwrap();
    let resolver = BuiltinOpResolver::default();

    let builder = InterpreterBuilder::new(&model, &resolver).unwrap();
    builder
}

fn main() {
    let info = rascam::info().unwrap();
    let mut camera = SimpleCamera::new(info.cameras[0].clone()).unwrap();
    camera.activate().unwrap();

    let sleep_duration = time::Duration::from_millis(2000);
    thread::sleep(sleep_duration);

    let b = camera.take_one().unwrap();
    File::create("image1.jpg").unwrap().write_all(&b).unwrap();
}
