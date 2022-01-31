#include <jack/jack.h>
#include <jack/midiport.h>
#include <jack/ringbuffer.h>
#include "backend.h"
#include "global.h"

class Backend_jack final : public Backend {
private:
    jack_client_t *client;
    const char **ports;
	const char *client_name = "timecannon";
	const char *server_name = NULL;
	jack_options_t options = JackNullOption;
	jack_status_t status;
    jack_port_t *input_port_L;
    jack_port_t *input_port_R;
    jack_port_t *output_port_L;
    jack_port_t *output_port_R;
    jack_port_t *input_port_MIDI;
    jack_port_t *output_port_MIDI;

    void (*external_process_callback)(Audio_buffers b, uint32_t nframes) = nullptr;
    void (*external_shutdown_callback)(void) = nullptr;

    static int internal_process_callback(uint32_t nframes, void *arg) const {
        Audio_buffers ap;
    	ap.inL  = (float *)jack_port_get_buffer (input_port_L, nframes);
        ap.inR  = (float *)jack_port_get_buffer (input_port_R, nframes);
        ap.outL = (float *)jack_port_get_buffer (output_port_L, nframes);
        ap.outR = (float *)jack_port_get_buffer (output_port_R, nframes);

        external_process_callback(ap, nframes);

        return 0;
    }
    static void internal_shutdown_callback(){
        external_shutdown_callback();
        jack_client_close(client);
        exit (1);
    }

public:
    Backend_jack(){}
    ~Backend_jack(){internal_shutdown_callback();}
    void sr_callback(){
        //call user defined callback - recalculate samplerate dependent processes. 
        sample_rate = jack_get_sample_rate(client);
    }
    void bs_callback(){ 
         //call user defined callback - recalculate blocksize dependent processes.
        block_size = jack_get_buffer_size(client);
    }
    void init_sr_bs() override {
        block_size = jack_get_buffer_size(client);
        sample_rate = jack_get_sample_rate(client);
        jack_set_buffer_size_callback(void(*bs_callback)());
        jack_set_sample_rate_callback(void(*sr_callback)());
    } 
    void initialize() override {
        client = jack_client_open (client_name, options, &status, server_name);
        if (client == NULL) {
            printf("jack_client_open() failed, "
                "status = 0x%2.0x\n", status);
            if (status & JackServerFailed) {
                printf("Unable to connect to JACK server\n");
            }
            exit (1);
        }
        if (status & JackServerStarted) {
            printf("JACK server started\n");
        }
        if (status & JackNameNotUnique) {
            client_name = jack_get_client_name(client);
            printf("unique name `%s' assigned\n", client_name);
        }
    }
    void prepare_to_play() override {
        input_port_MIDI  = jack_port_register (client, "midi_in", JACK_DEFAULT_MIDI_TYPE, JackPortIsInput, 0);
        output_port_MIDI = jack_port_register (client, "midi_out", JACK_DEFAULT_MIDI_TYPE, JackPortIsOutput, 0);

        input_port_L = jack_port_register (client, "inputL", JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);
        input_port_R = jack_port_register (client, "inputR", JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);

        output_port_L = jack_port_register (client, "outputL", JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
        output_port_R = jack_port_register (client, "outputR", JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
        
        if ((input_port_L == NULL) || (input_port_R == NULL) || (output_port_L == NULL)) {
            printf("no more JACK ports available\n");
            exit (1);
	    }

        if((sr != jack_get_sample_rate(client))||(blen != jack_get_buffer_size(client))){
            printf("sample rate or buffer size do not match");
            exit(1);
	    }   
    }
    void activate() override {
        if (jack_activate (client)) {
            printf("cannot activate client");
            exit (1);
        }
    /*
        Auto-connect the ports.  You can't do this before the client is
        activated, because we can't make connections to clients
        that aren't running.  Note the confusing (but necessary)
        orientation of the driver backend ports: playback ports are
        "input" to the backend, and capture ports are "output" from it.
    */

        ports = jack_get_ports (client, NULL, NULL,	JackPortIsPhysical|JackPortIsOutput);
        if (ports == NULL) {
            printf("no physical capture ports\n");
            exit (1);
        }
        if (jack_connect (client, ports[0], jack_port_name (input_port_L))) {
            printf("cannot connect input port L\n");
        }
        if (jack_connect (client, ports[1], jack_port_name (input_port_R))) {
            printf("cannot connect input port R\n");
        }
        if (jack_connect (client, ports[2], jack_port_name (input_port_MIDI))) {
            printf("cannot connect input port MIDI\n");
        }
        free (ports);
        
        ports = jack_get_ports (client, NULL, NULL,	JackPortIsPhysical|JackPortIsInput);
        if (ports == NULL) {
            printf("no physical playback ports\n");
            exit (1);
        }
        if (jack_connect (client, jack_port_name (output_port_L), ports[0])) {
            printf("cannot connect output port L\n");
        }
        if (jack_connect (client, jack_port_name (output_port_R), ports[1])) {
            printf("cannot connect output port R\n");
        }
        if (jack_connect (client, jack_port_name (output_port_MIDI), ports[2])) {
            printf("cannot connect output port MIDI\n");
        }
        free (ports);
    }
    void register_callback_process(void (*process_callback)(Audio_buffers b, uint32_t nframes)) override {
        external_process_callback = process_callback;
        jack_set_process_callback(client, internal_process_callback, 0);
    }
    void register_callback_shutdown(void (*shutdown_callback)(void)) override {
        external_shutdown_callback = shutdown_callback;
        jack_on_shutdown(client, internal_shutdown_callback, 0);
    }
    void register_callback_samplerate() override {}
}