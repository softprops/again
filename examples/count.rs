use std::{rc::Rc, sync::Mutex};

#[tokio::main]
async fn main() -> Result<(), &'static str> {
    pretty_env_logger::init();
    let counter = Rc::new(Mutex::new(0 as usize));
    again::retry(move || {
        let counter = counter.clone();
        async move {
            let mut num = counter.lock().unwrap();
            if *num > 1 {
                Ok(true)
            } else {
                *num += 1;
                Err("nope")
            }
        }
    })
    .await?;
    Ok(())
}
