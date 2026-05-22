fn main() {
    for (i, arg) in std::env::args().enumerate() {
        println!("arg {}: {}", i, arg);
    }
}
