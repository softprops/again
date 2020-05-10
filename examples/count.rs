use refut::Policy;
use std::{rc::Rc, sync::Mutex, time::Duration};

#[tokio::main]
async fn main() -> Result<(), String> {
    pretty_env_logger::init();
    let counter = Rc::new(Mutex::new(0 as usize));
    Policy::fixed(Duration::from_secs(1))
        .with_max_retries(10)
        .retry(move || {
            let counter = counter.clone();
            async move {
                let mut num = counter.lock().unwrap();
                if *num > 5 {
                    Ok(*num)
                } else {
                    *num += 1;
                    Err(format!("{} was too low try again", *num - 1))
                }
            }
        })
        .await?;
    Ok(())
}
