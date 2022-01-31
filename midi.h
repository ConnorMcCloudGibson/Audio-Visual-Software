//midi in handler, will do midi out later

/*


in midi handler class:
read raw byte
if it's a status byte:
    if it's a system message: act immediately (for single-byte msgs) or update state accordingly ... then break and wait

    if it's a channel message: update state (status + channel) accordingly ... then break and wait

if it's a data byte:
    (later i'll include 'inhibit' for sysex chunks)
    put into holding buffer, update 'bytes remaining'

if 'bytes remaining' = 0:
    call appropriate event handlers with data (+channel) as parameters
    (for now i can just do the implementation directly)


*/


struct midiIn {

    uint32_t bytes_remaining = 0;
    uint32_t notes_active = 0;
    uint32_t data[2] = {0};
    uint32_t state = 0;
    uint32_t channel = 0;

    void operator()(uint8_t rawbyte){

        // system common/RT... process then return;
        if(rawbyte >= 0xf0){
            switch(rawbyte & 0xf) {
                default:
                break;

                //COMMON, have to manage state
                case 0x0:
                // sysex, [ID, data, EOX] ... can be interrupted by system RT msgs
                state = 0x0;
                break;

                case 0x1: // MTC quarter frame
                state = 0x1;
                bytes_remaining = 1;
                break;

                case 0x2: // song position pointer
                state = 0x2;
                bytes_remaining = 2;
                break;

                case 0x3: // song select
                state = 0x3;
                bytes_remaining = 1;
                break;

                case 0x7:
                // EOX
                break;

                // RT, all single byte, don't alter state
                case 0x8:
                // clock
                break;
                case 0xa:
                // start
                break;
                case 0xb:
                // continue
                break;
                case 0xc:
                // stop
                break;
                case 0xe:
                // active sensing
                break;

            }
            return;
        }

        // channel messages
        // can do a check for whether the message is on active channel, but for now i'm in omni mode
        if(rawbyte >= 0x80){
            switch(rawbyte >> 4) {
                default:
                break;

                case 0x8: // note off - [note num][release vel]
                state = 0x8;
                bytes_remaining = 2;
                break;

                case 0x9: // note on - [note num][vel]
                state = 0x9;
                bytes_remaining = 2;
                break;

                case 0xa: // polyAT - [note num][pressure]
                state = 0xa;
                bytes_remaining = 2;
                break;

                case 0xb: // CC - [controller number][value]
                state = 0xb;
                bytes_remaining = 2;
                break;

                case 0xc: // program change - [program]
                state = 0xc;
                bytes_remaining = 1;
                break;

                case 0xd: // channel AT - [pressure]
                state = 0xd;
                bytes_remaining = 1;
                break;

                case 0xe: // PB - [LSB][MSB]
                state = 0xe;
                bytes_remaining = 2;
                break;

            }
            return;
        }



        if(rawbyte < 0x80){ // not strictly necessary
        	bytes_remaining--;
            data[bytes_remaining] = rawbyte;

            // note the data will be in reverse order in holding array


        }

        if(bytes_remaining == 0){ // if message is complete
        	
            switch(state) {
                default:

                break;

                case 0x0:
                // sysex, [ID, data, EOX] ... can be interrupted by system RT msgs
                break;
                case 0x1: // MTC quarter frame
                bytes_remaining = 1;
                break;
                case 0x2: // song position pointer
                bytes_remaining = 2;
                break;
                case 0x3: // song select
                bytes_remaining = 1;
                break;
                case 0x7:
                // EOX
                break;

                case 0x8: // note off - [note num][release vel]
                bytes_remaining = 2;
                // if(notes_active-- == 0){cv2 = 0;}
                break;

                case 0x9: // note on - [note num][vel]
                bytes_remaining = 2;
/*
                if(data[0] == 0){
                    if(--notes_active == 0){vel = 0.0f;}
                } else {
                    notes_active++;
                    note = data[1]/64.0f-1.0f; // slew? pos/neg? PB amount?
                    vel = data[0]/127.0f; // then do env follower in audio loop
                }
*/
                break;

                case 0xa: // polyAT - [note num][pressure]
                bytes_remaining = 2;
                break;

                case 0xb: // CC - [controller number][value]
                bytes_remaining = 2;
                break;

                case 0xc: // program change - [program]
                bytes_remaining = 1;
                break;

                case 0xd: // channel AT - [pressure]
                bytes_remaining = 1;
                break;

                case 0xe: // PB - [LSB][MSB]
                bytes_remaining = 2;
                break;

            }
            return;
        }


    }


};


/*

then.. in implementation

void midiIn::noteOn(channel, note, vel){

}

eh firstly just do it like copy paste...
all in one place

*/