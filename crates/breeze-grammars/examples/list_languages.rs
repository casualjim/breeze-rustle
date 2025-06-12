use breeze_grammars::{get_language, supported_languages, is_language_supported};

fn main() {
    println!("Supported languages:");
    for lang in supported_languages() {
        println!("  - {}", lang);
    }
    
    println!("\nTesting Python support:");
    println!("  is_language_supported(\"python\"): {}", is_language_supported("python"));
    println!("  is_language_supported(\"Python\"): {}", is_language_supported("Python"));
    
    if let Some(lang) = get_language("python") {
        println!("  Python language loaded successfully!");
        println!("  ABI version: {}", lang.abi_version());
    } else {
        println!("  Failed to load Python language");
    }
}