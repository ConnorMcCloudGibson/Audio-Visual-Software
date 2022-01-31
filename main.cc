





int main (int argc, char *argv[]){

// declarations:
    // audio backend.. fetch sample rate/block size from backend?
        // how to pass this info to other sections? globals
    // sequencer
    // audio
    // gui


// setup
    // backend.init() - 
    // backend.register_process_callback(audio.process)
    // register other callbacks: shutdown, samplerate/block size changes
    // backend.prepare_to_play() - register ports
    
    // setup anything for IPC - ringbuffers, initialising atomics etc

    // sequencer setup - load savefile
    // audio setup - including engine reset/init
    
    // gui setup

    // visuals setup - windowing/GL context, compile shaders

    // backend activate

    //while(1){gui / visuals}
    // event loop + interleaving visuals/gui draw calls

    //shutdown();
    
}



// functions
    // shutdown






/*

[static] globals...
performance penalty of reference member variables bound at runtime
constexpr constructors
could declare in any order and then have bind methods in case bidirectional binding needs to happen

*/

