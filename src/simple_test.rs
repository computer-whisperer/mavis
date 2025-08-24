use candle_core::Device;
use candle_core::utils::cuda_is_available;
use candle_transformers::generation::LogitsProcessor;
use crate::text_generation::TextGeneration;

mod text_generation;
mod parler_text_to_speech;
mod llama;
mod hybrid_var_map;

fn main() {

    let device = if cuda_is_available() {
        Device::new_cuda(0).unwrap()
    }
    else {
        Device::Cpu
    };

    let test_context_string = "[computer-whisperer]: data to the
[computer-whisperer]: chatbot thread
[computer-whisperer]: I'm attributing these crashes at the moment to...
[computer-whisperer]: The speech-to-text thread blocks for on the order of a second or so whenever it runs a
[computer-whisperer]: A batch?
[computer-whisperer]: It's like 10 seconds of data or so that it runs in much faster than real time. But still it's enough time that...
[computer-whisperer]: That was blocking...
[u6bkep]: *Grunts*
[computer-whisperer]: up for just long enough for the opus packets to pile up from the UDP thread.
[computer-whisperer]: Because they're apparently being delivered in...
[computer-whisperer]: There are a lot of Opus packets, and they are usually quite small.
[computer-whisperer]: So that, even though I had a...
[computer-whisperer]: 256 entry queue there.
[computer-whisperer]: for storing Opus packets, it was not enough.
[computer-whisperer]: So now there's a separate thread designed to take the Opus packets, decode them, batch them into...
[computer-whisperer]: larger PCM buffers.
[computer-whisperer]: and then submit a 10-second PCM buffer at a time to the text-to-speech.
[computer-whisperer]: for speech-to-text.
[u6bkep]: Yeah, I think standard practice is to turn the audio packet size on mumble down to 10 milliseconds.
[computer-whisperer]: Yeah, that's a lot.
[computer-whisperer]: Now let's wait for tomorrow's gaming session to try it again, because it's...
[computer-whisperer]: It's really the multiple people talking for a long period of time and triggering lots of speech attacks simultaneously that really does it.
[u6bkep]: Do we need to spin up the, uh...
[u6bkep]: A little cluster of mobile clients.
[computer-whisperer]: I don't know. Oh, you know what? I want, uh...
[DondeathOnPC]: Well, that's kind of terrifying, now isn't it?
[computer-whisperer]: You said...
[computer-whisperer]: Okay, uh...
[computer-whisperer]: There is solid evidence that Matt is triggering these crashes.
[u6bkep]: Mmm.
[computer-whisperer]: Uh, I wonder what his packet size is set to.
[u6bkep]: Hmmmmmm
[computer-whisperer]: Also, it may be a lot better now, so, uh...
[computer-whisperer]: Might have fixed it. We'll see.
[u6bkep]: The mumble command line
[mavis]: there.
[u6bkep]: Help has a specific note about how to run multiple versions of mumble at once.
[u6bkep]: or multiple instances a month.
[DondeathOnPC]: Is it more detailed than \"Don't do this, what is wrong with you\"?
[u6bkep]: No, it says if you want to do that, don't forget to change these other config values as well.
[computer-whisperer]: I mean, I know how to be a mumble client now.
[computer-whisperer]: Beavis is just a...
[computer-whisperer]: A mumble client with a lot of extra things tacked on.
[SERVER]: DISCONNECTED FROM SERVER AT 2025-01-25 04:38:13.935566719 UTC
[SERVER]: CONNECTED TO SERVER AT 2025-01-26 01:15:03.136513667 UTC
[SERVER]: Entered channel Root of the gecko tree
[SERVER]: thefiregecko present
[SERVER]: computer-whisperer present
[SERVER]: u6bkep present
[computer-whisperer]: hello mavis
";

    let test_train_text = "
[mavis]: there.
[u6bkep]: Help has a specific note about how to run multiple versions of mumble at once.
[u6bkep]: or multiple instances a month.
[DondeathOnPC]: Is it more detailed than \"Don't do this, what is wrong with you\"?
[u6bkep]: No, it says if you want to do that, don't forget to change these other config values as well.
[computer-whisperer]: I mean, I know how to be a mumble client now.
[computer-whisperer]: Beavis is just a...
[computer-whisperer]: A mumble client with a lot of extra things tacked on.
[SERVER]: DISCONNECTED FROM SERVER AT 2025-01-25 04:38:13.935566719 UTC
[SERVER]: CONNECTED TO SERVER AT 2025-01-26 01:15:03.136513667 UTC
[SERVER]: Entered channel Root of the gecko tree
[SERVER]: thefiregecko present
[SERVER]: computer-whisperer present
[SERVER]: u6bkep present
[computer-whisperer]: hello mavis
";

    if true {
        let mut text_generation = TextGeneration::new(device.clone(), None).unwrap();
        println!("\n\n\n");
        let mut logits_processor = LogitsProcessor::new(299792458, Some(1.0), None);
        let mut context = text_generation.new_context().unwrap();
        context.add_unprocessed_text(test_context_string);
        for _ in 0..1 {
            text_generation.train_text(Some(&mut context), test_train_text, 0.0001).unwrap();
        }
        let mut context = text_generation.new_context().unwrap();
        context.add_unprocessed_text("This is a long");
        println!("Generated text: {}", text_generation.run(&mut context, &mut logits_processor).unwrap());

        context = text_generation.new_context().unwrap();
        context.add_unprocessed_text("1 2 3 4");
        println!("Generated text: {}", text_generation.run(&mut context, &mut logits_processor).unwrap());

        return;
    }

    let mut tts = parler_text_to_speech::ParlerTextToSpeech::new(device).unwrap();
    tts.single_shot("Hello, world!");
}
