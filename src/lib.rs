//! `Again` is a wasm-compatible utility for retrying standard library [`Futures`](https://doc.rust-lang.org/std/future/trait.Future.html) with a `Result` output type
//!
//! A goal of any operation should be a successful outcome. This crate gives operations a better chance at achieving that.
//!
//! # Examples
//!
//! ## Hello world
//!
//! For simple cases, you can use the module level [`retry`](fn.retry.html) fn, which
//! will retry a task every second for 5 seconds with an exponential backoff.
//!
//! ```no_run
//! again::retry(|| reqwest::get("https://api.company.com"));
//! ```
//!
//! ## Conditional retries
//!
//! By default, `again` will retry any failed `Future` if its `Result` output type is an `Err`.
//! You may not want to retry _every_ kind of error. In those cases you may wish to use the [`retry_if`](fn.retry_if.html) fn, which
//! accepts an additional argument to conditionally determine if the error
//! should be retried.
//!
//! ```no_run
//! again::retry_if(
//!     || reqwest::get("https://api.company.com"),
//!     reqwest::Error::is_status,
//! );
//! ```
//!
//! ## Retry policies
//!
//! Every application has different needs. The default retry behavior in `again`
//! likely will not suit all of them. You can define your own retry behavior
//! with a [`RetryPolicy`](struct.RetryPolicy.html). A RetryPolicy can be configured with a fixed or exponential backoff,
//! jitter, and other common retry options. This objects may be reused
//! across operations. For more information see the [`RetryPolicy`](struct.RetryPolicy.html) docs.
//!
//! ```no_run
//! use again::RetryPolicy;
//! use std::time::Duration;
//!
//! let policy = RetryPolicy::fixed(Duration::from_millis(100))
//!     .with_max_retries(10)
//!     .with_jitter(false);
//!
//! policy.retry(|| reqwest::get("https://api.company.com"));
//! ```
//!
//! # Logging
//!
//! For visibility on when operations fail and are retired, a `log::trace` message is emitted,
//! logging the `Debug` display of the error and the delay before the next attempt.
//!
//! # wasm
//!
//! `again` supports [WebAssembly](https://webassembly.org/) targets i.e. `wasm32-unknown-unknown` which should make this
//! crate a good fit for most environments
use rand::{distributions::OpenClosed01, thread_rng, Rng};
use std::{cmp::min, future::Future, time::Duration};
use wasm_timer::Delay;

/// Retries a fallible `Future` with a default `RetryPolicy`
///
/// ```
/// again::retry(|| async { Ok::<u32, ()>(42) });
/// ```
pub async fn retry<T>(task: T) -> Result<T::Item, T::Error>
where
    T: Task,
{
    crate::retry_if(task, Always).await
}

/// Retries a fallible `Future` under a certain provided conditions with a default `RetryPolicy`
///
/// ```
/// again::retry_if(|| async { Err::<u32, u32>(7) }, |err: &u32| *err != 42);
/// ```
pub async fn retry_if<T, C>(
    task: T,
    condition: C,
) -> Result<T::Item, T::Error>
where
    T: Task,
    C: Condition<T::Error>,
{
    RetryPolicy::default().retry_if(task, condition).await
}

#[derive(Clone, Copy)]
enum Backoff {
    Fixed,
    Exponential,
}

impl Default for Backoff {
    fn default() -> Self {
        Backoff::Exponential
    }
}

impl Backoff {
    fn iter(
        self,
        policy: &RetryPolicy,
    ) -> BackoffIter {
        BackoffIter {
            backoff: self,
            current: 1,
            jitter: policy.jitter,
            delay: policy.delay,
            max_delay: policy.max_delay,
            max_retries: policy.max_retries,
        }
    }
}

struct BackoffIter {
    backoff: Backoff,
    current: u32,
    jitter: bool,
    delay: Duration,
    max_delay: Option<Duration>,
    max_retries: usize,
}

impl Iterator for BackoffIter {
    type Item = Duration;
    fn next(&mut self) -> Option<Self::Item> {
        if self.max_retries > 0 {
            let factor = match self.backoff {
                Backoff::Fixed => Some(self.current),
                Backoff::Exponential => {
                    let factor = self.current;
                    if let Some(next) = self.current.checked_mul(2) {
                        self.current = next;
                    } else {
                        self.current = u32::MAX;
                    }

                    Some(factor)
                }
            };

            if let Some(factor) = factor {
                if let Some(mut delay) = self.delay.checked_mul(factor) {
                    if self.jitter {
                        delay = jitter(delay);
                    }
                    if let Some(max_delay) = self.max_delay {
                        delay = min(delay, max_delay);
                    }
                    self.max_retries -= 1;
                    return Some(delay);
                }
            }
        }
        None
    }
}

/// A template for configuring retry behavior
///
/// A default is provided, configured
/// to retry a task 5 times with exponential backoff
/// starting with a 1 second delay
#[derive(Clone)]
pub struct RetryPolicy {
    backoff: Backoff,
    jitter: bool,
    delay: Duration,
    max_delay: Option<Duration>,
    max_retries: usize,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            backoff: Backoff::default(),
            delay: Duration::from_secs(1),
            jitter: false,
            max_delay: None,
            max_retries: 5,
        }
    }
}

fn jitter(duration: Duration) -> Duration {
    let jitter: f64 = thread_rng().sample(OpenClosed01);
    let secs = (duration.as_secs() as f64) * jitter;
    let nanos = (duration.subsec_nanos() as f64) * jitter;
    let millis = (secs * 1_000_f64) + (nanos / 1_000_000_f64);
    Duration::from_millis(millis as u64)
}

impl RetryPolicy {
    fn backoffs(&self) -> impl Iterator<Item = Duration> {
        self.backoff.iter(self)
    }

    /// Configures policy with an exponential
    /// backoff delay.
    ///
    /// By default, Futures will be retied 5 times.
    ///
    /// These delays will increase in
    /// length over time. You may wish to cap just how long
    /// using the [`with_max_delay`](struct.Policy.html#method.with_max_delay) fn
    pub fn exponential(delay: Duration) -> Self {
        Self {
            backoff: Backoff::Exponential,
            delay,
            ..Self::default()
        }
    }

    /// Configures policy with a fixed
    /// backoff delay.
    ///
    /// By default, Futures will be retied 5 times.
    ///
    /// These delays will increase in
    /// length over time. You may wish to configure how many
    /// times a Future will be retired using the [`with_max_retries`](struct.RetryPolicy.html#method.with_max_retries) fn    
    pub fn fixed(delay: Duration) -> Self {
        Self {
            backoff: Backoff::Fixed,
            delay,
            ..Self::default()
        }
    }

    /// Configures randomness to the delay between retries.
    ///
    /// This is useful for services that have many clients which might all retry at the same time to avoid
    /// the ["thundering herd" problem](https://en.wikipedia.org/wiki/Thundering_herd_problem)
    pub fn with_jitter(
        mut self,
        jitter: bool,
    ) -> Self {
        self.jitter = jitter;
        self
    }

    /// Limits the maximum length of delay between retries
    pub fn with_max_delay(
        mut self,
        max: Duration,
    ) -> Self {
        self.max_delay = Some(max);
        self
    }

    /// Limits the maximum number of attempts a Future will be tried
    pub fn with_max_retries(
        mut self,
        max: usize,
    ) -> Self {
        self.max_retries = max;
        self
    }

    /// Retries a fallible `Future` with this policy's configuration
    pub async fn retry<T>(
        &self,
        task: T,
    ) -> Result<T::Item, T::Error>
    where
        T: Task,
    {
        self.retry_if(task, Always).await
    }

    /// Retries a fallible `Future` with this policy's configuration under certain provided conditions
    pub async fn retry_if<T, C>(
        &self,
        task: T,
        condition: C,
    ) -> Result<T::Item, T::Error>
    where
        T: Task,
        C: Condition<T::Error>,
    {
        let mut backoffs = self.backoffs();
        let mut task = task;
        let mut condition = condition;
        loop {
            match task.call().await {
                Ok(result) => return Ok(result),
                Err(err) => {
                    if condition.is_retryable(&err) {
                        if let Some(delay) = backoffs.next() {
                            log::trace!(
                                "task failed with error {:?}. will try again in {:?}",
                                err,
                                delay
                            );
                            let _ = Delay::new(delay).await;
                            continue;
                        }
                    }
                    return Err(err);
                }
            }
        }
    }
}

/// A type to determine if a failed Future should be retried
///
/// A implementation is provided for `Fn(&Err) -> bool` allowing you
/// to use simple closure or fn handles
pub trait Condition<E> {
    /// Return true if an Futures error is worth retrying
    fn is_retryable(
        &mut self,
        error: &E,
    ) -> bool;
}

struct Always;

impl<E> Condition<E> for Always {
    #[inline]
    fn is_retryable(
        &mut self,
        _: &E,
    ) -> bool {
        true
    }
}

impl<F, E> Condition<E> for F
where
    F: Fn(&E) -> bool,
{
    fn is_retryable(
        &mut self,
        error: &E,
    ) -> bool {
        self(error)
    }
}

/// A unit of work to be retried
///
/// A implementation is provided for `FnMut() -> Future`
pub trait Task {
    /// The `Ok` variant of a `Futures` associated Output type
    type Item;
    /// The `Err` variant of `Futures` associated Output type
    type Error: std::fmt::Debug;
    /// The resulting `Future` type
    type Fut: Future<Output = Result<Self::Item, Self::Error>>;
    /// Call the operation which invokes results in a `Future`
    fn call(&mut self) -> Self::Fut;
}

impl<F, Fut, I, E> Task for F
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<I, E>>,
    E: std::fmt::Debug,
{
    type Item = I;
    type Error = E;
    type Fut = Fut;

    fn call(&mut self) -> Self::Fut {
        self()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn retry_policy_is_send() {
        fn test(_: impl Send) {}
        test(RetryPolicy::default())
    }

    #[test]
    fn jitter_adds_variance_to_durations() {
        assert!(jitter(Duration::from_secs(1)) != Duration::from_secs(1));
    }

    #[test]
    fn backoff_default() {
        assert!(matches!(Backoff::default(), Backoff::Exponential));
    }

    #[test]
    fn fixed_backoff() {
        let mut iter = RetryPolicy::fixed(Duration::from_secs(1)).backoffs();
        assert_eq!(iter.next(), Some(Duration::from_secs(1)));
        assert_eq!(iter.next(), Some(Duration::from_secs(1)));
        assert_eq!(iter.next(), Some(Duration::from_secs(1)));
        assert_eq!(iter.next(), Some(Duration::from_secs(1)));
    }

    #[test]
    fn exponential_backoff() {
        let mut iter = RetryPolicy::exponential(Duration::from_secs(1)).backoffs();
        assert_eq!(iter.next(), Some(Duration::from_secs(1)));
        assert_eq!(iter.next(), Some(Duration::from_secs(2)));
        assert_eq!(iter.next(), Some(Duration::from_secs(4)));
        assert_eq!(iter.next(), Some(Duration::from_secs(8)));
    }

    #[test]
    fn always_is_always_retryable() {
        assert!(Always.is_retryable(&()))
    }

    #[test]
    fn closures_impl_condition() {
        fn test(_: impl Condition<()>) {}
        #[allow(clippy::trivially_copy_pass_by_ref)]
        fn foo(_err: &()) -> bool {
            true
        }
        test(foo);
        test(|_err: &()| true);
    }

    #[test]
    fn closures_impl_task() {
        fn test(_: impl Task) {}
        async fn foo() -> Result<u32, ()> {
            Ok(42)
        }
        test(foo);
        test(|| async { Ok::<u32, ()>(42) });
    }

    #[test]
    fn retied_futures_are_send_when_tasks_are_send() {
        fn test(_: impl Send) {}
        test(RetryPolicy::default().retry(|| async { Ok::<u32, ()>(42) }))
    }

    #[tokio::test]
    async fn ok_futures_yield_ok() -> Result<(), Box<dyn Error>> {
        let result = RetryPolicy::default()
            .retry(|| async { Ok::<u32, ()>(42) })
            .await;
        assert_eq!(result, Ok(42));
        Ok(())
    }

    #[tokio::test]
    async fn failed_futures_yield_err() -> Result<(), Box<dyn Error>> {
        let result = RetryPolicy::fixed(Duration::from_millis(1))
            .retry(|| async { Err::<u32, ()>(()) })
            .await;
        assert_eq!(result, Err(()));
        Ok(())
    }
}
