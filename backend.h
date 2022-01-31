class Backend  {
private:
    static uint32_t block_size = 0;
    static uint32_t sample_rate = 0;
    // audio.process function ptr
    // internal process callback which formats buffers/midi then calls the above
public:
    Backend(){init_sr_bs();}
    ~Backend(){}
    virtual void init_sr_bs() = 0; // set sample rate and block size
    virtual void initialize() = 0;
    virtual void prepare_to_play() = 0;
    virtual void activate() = 0;
    virtual void register_callback_process(void (*process)(Audio_buffers b)) = 0;
    virtual void register_callback_shutdown() = 0;
    virtual void register_callback_samplerate() = 0;
}

// MIDI?
// MIDI API - not sent via process() argument, but using tx/rx functions within process()
// get_next_midi_event
// format messages or raw bytes?
// use separate midi.h library for msg decode etc
// setup functionality - port, channel
// register 'callbacks' - note on/off, midi CC, 