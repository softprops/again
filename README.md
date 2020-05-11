<h1 align="center">
  again ♻️
</h1>

<p align="center">
   retry interfaces for fallible Rustlang std library Futures
</p>

<div align="center">
  <a href="https://github.com/softprops/again/actions">
		<img src="https://github.com/softprops/again/workflows/Main/badge.svg"/>
	</a>
  <a href="https://crates.io/crates/again">
		<img src="http://meritbadge.herokuapp.com/again"/>
	</a>
  <a href="http://docs.rs/again">
		<img src="https://docs.rs/again/badge.svg"/>
	</a>  
  <a href="https://softprops.github.io/again">
		<img src="https://img.shields.io/badge/docs-master-green.svg"/>
	</a>
</div>

<br />

A goal of any operation should be a successful outcome. This crate gives operations a better chance at achieving that.

## 📦 install

In your Cargo.toml file, add the following under the `[dependencies]` heading

```toml
again = "0.1"
```

## 🤸usage

For very simple cases you can use the module level `retry` function
to retry a potentially fallible operation.

```rust
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    pretty_env_logger::init();
    again::retry(|| reqwest::get("https://api.you.com")).await?;
    Ok(())
}
```

You can also customize retry behavior to suit your applications needs
with a reusable `RetryPolicy`.

```rust
use std::error::Error;
use std::time::Duration;
use again::RetryPolicy;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    pretty_env_logger::init();
    let policy = RetryPolicy.exponential(Duration::from_millis(200))
      .with_max_retries(10)
      .with_jitter(true);
    again::retry(|| reqwest::get("https://api.you.com")).await?;
    Ok(())
}
```

See the [docs](http://docs.rs/again) for more examples.

Doug Tangren (softprops) 2020