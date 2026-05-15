use anyhow::Result;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

pub trait EventProducer<I, O>: Sized {
    fn spawn(self, input: mpsc::Receiver<I>, output: mpsc::Sender<O>) -> JoinHandle<Result<()>>;
}

pub struct EventProducerChannels<I, O> {
    pub input: mpsc::Sender<I>,
    pub output: mpsc::Receiver<O>,
    pub task: JoinHandle<Result<()>>,
}

pub fn spawn_event_channels<P, I, O>(
    producer: P,
    input_capacity: usize,
    output_capacity: usize,
) -> EventProducerChannels<I, O>
where
    P: EventProducer<I, O>,
{
    let (input_tx, input_rx) = mpsc::channel(input_capacity);
    let (output_tx, output_rx) = mpsc::channel(output_capacity);
    let task = producer.spawn(input_rx, output_tx);
    EventProducerChannels {
        input: input_tx,
        output: output_rx,
        task,
    }
}
