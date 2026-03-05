import torch
import copy
import json

import torch
from torch.utils.data import Dataset
import glob
import mido
from collections import defaultdict
import json
from concurrent.futures import ProcessPoolExecutor
import tqdm
import numpy as np
import pandas as pd
import os

def pitch_to_octave_pitch_class(pitch):
    return pitch//12, pitch%12

def octave_pitch_class_to_pitch(octave, pitch_class):
    return int(octave*12+pitch_class)

def pitch_to_octave_pitch_class_microtonal(canonical_pitch_semitones):
    """Float semitones -> (octave, pitch_class_cents 0-1199)"""
    octave = int(canonical_pitch_semitones // 12)
    pitch_class_cents = int(round((canonical_pitch_semitones % 12) * 100))
    pitch_class_cents = max(0, min(1199, pitch_class_cents))
    return octave, pitch_class_cents

def octave_pitch_class_to_pitch_microtonal(octave, pitch_class_cents):
    """Reverse: (octave, cents) -> float semitones"""
    return octave * 12.0 + pitch_class_cents / 100.0

def pitchbend_to_semitones(pitchbend_value, sensitivity=2.0):
    """MIDI pitchbend int [-8192, 8191] -> semitone offset float"""
    return (pitchbend_value / 8192.0) * sensitivity

def canonical_pitch_to_midi_and_pitchbend(canonical_pitch_semitones, sensitivity=2.0):
    """Float semitones -> (midi_note 0-127, pitchbend int)"""
    midi_note = int(round(canonical_pitch_semitones))
    midi_note = max(0, min(127, midi_note))
    residual = canonical_pitch_semitones - midi_note
    pitchbend = int(round((residual / sensitivity) * 8192))
    pitchbend = max(-8192, min(8191, pitchbend))
    return midi_note, pitchbend


class MusicTokenizer():
    def __init__(self, timeshift_vocab_size = 21, dur_vocab_size = 1001, octave_vocab_size = 9, pitch_class_vocab_size = 12, instrument_vocab_size = 128, velocity_vocab_size = 129, sos_token = -1, eos_token = -2, pad_token = -3, microtonal=False, pitchbend_sensitivity=2.0):
        self.timeshift_vocab_size = timeshift_vocab_size
        self.dur_vocab_size = dur_vocab_size
        self.octave_vocab_size = octave_vocab_size
        self.pitch_class_vocab_size = pitch_class_vocab_size
        self.instrument_vocab_size = instrument_vocab_size
        self.velocity_vocab_size = velocity_vocab_size
        self.sos_out_vocab_size = 1 #sos token at the output side, only 1 token
        self.sos_out = 0
        self.microtonal = microtonal
        self.pitchbend_sensitivity = pitchbend_sensitivity

        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.sos_token_compound = [self.sos_token for _ in range(6)]
        self.eos_token_compound = [self.eos_token for _ in range(6)]
        self.pad_token_compound = [self.pad_token for _ in range(6)]

        #define labels (language tokens)
        self.sos_timeshift, self.eos_timeshift = self.timeshift_vocab_size-2, self.timeshift_vocab_size-1 #TODO think whether this is correct
        self.sos_dur, self.eos_dur = self.dur_vocab_size-2, self.dur_vocab_size-1
        # self.sos_dur, self.eos_dur = 1025, 1026 #TODO: very ugly fix!! #IF CONVERT TO LINEAR THEN HAVE TO CHANGE THIS BACK

        self.sos_octave, self.eos_octave = self.octave_vocab_size-2, self.octave_vocab_size-1
        self.sos_pitch_class, self.eos_pitch_class = self.pitch_class_vocab_size-2, self.pitch_class_vocab_size-1
        self.sos_instrument, self.eos_instrument = self.instrument_vocab_size-2, self.instrument_vocab_size-1
        self.sos_velocity, self.eos_velocity = self.velocity_vocab_size-2, self.velocity_vocab_size-1


        self.sos_label = [self.sos_out, self.sos_timeshift, self.sos_dur, self.sos_octave, self.sos_pitch_class, self.sos_instrument, self.sos_velocity]
        self.eos_label = [self.sos_out, self.eos_timeshift, self.eos_dur, self.eos_octave, self.eos_pitch_class, self.eos_instrument, self.eos_velocity]

        # self.onset_dict = {i: i for i in range(2)}
        self.sos_out_dict = {i: i for i in range(self.sos_out_vocab_size)}
        self.timeshift_dict = {i: i+self.sos_out_vocab_size for i in range(self.timeshift_vocab_size)}
        self.duration_dict = {i: i+self.sos_out_vocab_size+self.timeshift_vocab_size for i in range(self.dur_vocab_size)} #linear scale
        self.octave_dict = {i: i+self.sos_out_vocab_size+self.timeshift_vocab_size+self.dur_vocab_size for i in range(self.octave_vocab_size)}

        if self.microtonal:
            # Append-only pitch_dict: western cents -> original IDs, microtonal -> appended after 8487
            # For 10-cent ablation: change loop to iterate cent values 0,10,20,...,1190 instead of 0..1199.
            # Use: `for idx in range(vocab_size - 2): c = idx * resolution` with SOS=1200, EOS=1201.
            self._original_pitch_count = 14  # original 12 pitches + SOS + EOS
            pitch_offset = (self.sos_out_vocab_size + self.timeshift_vocab_size +
                            self.dur_vocab_size + self.octave_vocab_size)
            # pitch_offset = 8212 for default config
            original_decode_vocab = pitch_offset + self._original_pitch_count + self.instrument_vocab_size + self.velocity_vocab_size
            # original_decode_vocab = 8487

            self.pitch_dict = {}
            micro_id = original_decode_vocab  # start appending at 8487

            for c in range(self.pitch_class_vocab_size):  # 0..1201
                if c < 1200 and c % 100 == 0:
                    # Western semitone: map to ORIGINAL pretrained ID
                    western_pc = c // 100  # 0-11
                    self.pitch_dict[c] = pitch_offset + western_pc
                elif c >= 1200:
                    # SOS (1200) and EOS (1201): map to original SOS/EOS pitch IDs
                    original_idx = 12 + (c - 1200)  # 12 = SOS, 13 = EOS
                    self.pitch_dict[c] = pitch_offset + original_idx
                else:
                    # Microtonal cent: append after original vocab
                    self.pitch_dict[c] = micro_id
                    micro_id += 1
            # micro_id should now be 8487 + 1188 = 9675

            # Instrument and velocity keep ORIGINAL offsets (not shifted by pitch expansion)
            instr_offset = pitch_offset + self._original_pitch_count  # 8212 + 14 = 8226
            self.instrument_dict = {i: instr_offset + i for i in range(self.instrument_vocab_size)}
            vel_offset = instr_offset + self.instrument_vocab_size  # 8226 + 131 = 8357
            self.velocity_dict = {i: vel_offset + i for i in range(self.velocity_vocab_size)}
        else:
            self.pitch_dict = {i: i+self.sos_out_vocab_size+self.timeshift_vocab_size+self.dur_vocab_size+self.octave_vocab_size for i in range(self.pitch_class_vocab_size)}
            self.instrument_dict= {i: i+self.sos_out_vocab_size+self.timeshift_vocab_size+self.dur_vocab_size+self.octave_vocab_size+self.pitch_class_vocab_size for i in range(self.instrument_vocab_size)}
            self.velocity_dict = {i: i+self.sos_out_vocab_size+self.timeshift_vocab_size+self.dur_vocab_size+self.octave_vocab_size+self.pitch_class_vocab_size + self.instrument_vocab_size for i in range(self.velocity_vocab_size)}
        print(f"self.sos_out_dict:{self.sos_out_dict}, self.timeshift_dict:{self.timeshift_dict}, self.duration_dict:{self.duration_dict},self.octave_dict:{self.octave_dict}, self.pitch_dict (first 5):{dict(list(self.pitch_dict.items())[:5])}, self.instrument_dict (first 3):{dict(list(self.instrument_dict.items())[:3])} self.velocity_dict (first 3):{dict(list(self.velocity_dict.items())[:3])}")

        self.sos_out_dict_decode = {v: k for k, v in self.sos_out_dict.items()}
        self.timeshift_dict_decode = {v: k for k, v in self.timeshift_dict.items()}
        self.duration_dict_decode = {v: k for k, v in self.duration_dict.items()}
        self.octave_dict_decode = {v: k for k, v in self.octave_dict.items()}
        self.pitch_dict_decode = {v: k for k, v in self.pitch_dict.items()}
        self.instrument_dict_decode = {v: k for k, v in self.instrument_dict.items()}
        self.velocity_dict_decode = {v: k for k, v in self.velocity_dict.items()}

        # print(f"self.onset_dict_decode:{self.onset_dict_decode}, self.duration_dict_decode:{self.duration_dict_decode},self.octave_dict_decode:{self.octave_dict_decode}, self.pitch_dict_decode:{self.pitch_dict_decode}, self.instrument_dict_decode:{self.instrument_dict_decode} self.velocity_dict_decode:{self.velocity_dict_decode}")
    def encode_single(self, raw_token):
        """
        each raw token looks like: [onset_binary_str, duration, [octave, pitch_class],instrument, velocity], ['00000001101100111101', 25, [4, 11], 24, 58] 
        """

        onset = raw_token[0]

        duration = raw_token[1]

        octave = raw_token[2]

        pitch_class = raw_token[3]

        instrument = raw_token[4]

        velocity = raw_token[5]

        return [onset, duration, octave, pitch_class, instrument, velocity]

    def encode_series(self, raw_token_series, if_add_sos, if_add_eos):
        out = [self.encode_single(x) for x in raw_token_series]
        if if_add_sos:
            out = [self.sos_token_compound] + out
        if if_add_eos:
            out = out + [self.eos_token_compound]
        return out #archive: during training [0 for _ in range(6)] will become the "position" of the SOS token. EOS wont be position-encoded. 
    def encode_series_labels(self, encoded_tokens, if_added_sos, if_added_eos):
        encoded_tokens = torch.tensor(encoded_tokens) #temporarily convert to tensor for easier slicing (len, 6)

        if if_added_sos:
            encoded_tokens = encoded_tokens[1:]
        if if_added_eos:
            encoded_tokens = encoded_tokens[:-1]
        
        #now the encoded tokens does not contain sos and eos
        """
        #1. Convert Onset to bits 
        onsets_labels = self.decimal_to_binary_batch(encoded_tokens[:, 0], bits=self.onset_vocab_size) #(len, ) -> (len, onset_vocab_size)
        other_labels = encoded_tokens[:, 1:] #(len, 5)"""

        #1. Convert onsets to delta onsets
        # print(f"check encoded_tokens:{encoded_tokens[:5]}")
        timeshift_labels_raw =  torch.diff(encoded_tokens[:, 0], prepend=torch.tensor([0]))
        # print(f"timeshift_labels_raw:{timeshift_labels_raw[:5]}")
        #2. Concat the raw value
        """
        output = torch.concat([onsets_labels, other_labels], dim = -1) #(len, onset_vocab_size+5)"""
        output = torch.cat([torch.zeros(encoded_tokens.shape[0]).unsqueeze(-1) , timeshift_labels_raw.unsqueeze(-1), encoded_tokens[:, 1:]], dim = -1) #(len, 6) #TODO: check if this is correct, ensure there is no time shift < 0
        # print(f"output:{output[:5]}")

        """add sos_out token at the beginning"""
        #3. Add sos and eos label if necessary
        if if_added_sos: 
            output = torch.concat([torch.tensor(self.sos_label).unsqueeze(0), output], dim = 0) 
        if if_added_eos:
            output = torch.concat([output, torch.tensor(self.eos_label).unsqueeze(0)], dim = 0) 
        output = output.tolist() 

        #4. encode it to labels 
        labels = [self.convert_to_language_tokens(x) for x in output]
        return labels
    def encode_series_con_gen_emotion(self, raw_token_series, if_add_sos, if_add_eos, emotion_token_4Q = None):
        # Emo_4Q (total 4 types) + SOS + music_seq + EOS 
        out = [self.encode_single(x) for x in raw_token_series]
        if if_add_sos:
            out = [self.sos_token_compound] + out
        if emotion_token_4Q is not None:
            out = [emotion_token_4Q] + out
        if if_add_eos:
            out = out + [self.eos_token_compound]
        return out

    def encode_series_labels_con_gen_emotion(self, encoded_tokens, if_added_sos, if_added_eos, if_added_emotion_token_4Q = None):
        # Emo_4Q + SOS + music_seq + EOS 
        encoded_tokens = torch.tensor(encoded_tokens) #temporarily convert to tensor for easier slicing (len, 6)
        if if_added_emotion_token_4Q:
            encoded_tokens = encoded_tokens[1:]
        if if_added_sos:
            encoded_tokens = encoded_tokens[1:]
        if if_added_eos:
            encoded_tokens = encoded_tokens[:-1]

        #1. Convert onsets to delta onsets

        timeshift_labels_raw =  torch.diff(encoded_tokens[:, 0], prepend=torch.tensor([0]))
        #2. Concat the raw value

        output = torch.cat([torch.zeros(encoded_tokens.shape[0]).unsqueeze(-1) , timeshift_labels_raw.unsqueeze(-1), encoded_tokens[:, 1:]], dim = -1) 

        #3. Add sos and eos label if necessary
        if if_added_sos: 
            output = torch.concat([torch.tensor(self.sos_label).unsqueeze(0), output], dim = 0) 

        if if_added_emotion_token_4Q is not None: #emotion token should always followed by the sos token in this task, we can use dummy labels (self.sos_label) for emotion_token_4Q tokens.
            output = torch.concat([torch.tensor(self.sos_label).unsqueeze(0), output], dim = 0) 

        if if_added_eos:
            output = torch.concat([output, torch.tensor(self.eos_label).unsqueeze(0)], dim = 0) 
        output = output.tolist() 

        #4. encode it to labels 
        labels = [self.convert_to_language_tokens(x) for x in output]
        return labels

    def encode_series_con_gen_commu(self, raw_token_series, raw_chord_series, metadata_tokens, if_only_keep_condition_tokens = False, if_add_chords_in_transformer = True, if_add_metadata_in_transformer = False):
        # meta_data_tokens,<SOC> chords, <EOC>, <SOS> music_seq, <EOS>   
        if not if_only_keep_condition_tokens:
            out = [self.encode_single(x) for x in raw_token_series]
            out = [self.sos_token_compound] + out + [self.eos_token_compound]
        else:
            out = []

        if if_add_chords_in_transformer:
            out_chord = [self.encode_single(x) for x in raw_chord_series]
            out_chord = [self.soc_token_compound] + out_chord + [self.eoc_token_compound]
        else:
            out_chord = []

        if if_add_metadata_in_transformer:
            metadata_tokens = metadata_tokens
        else:
            metadata_tokens = []
            


        if if_only_keep_condition_tokens: #Used during inference
            return metadata_tokens + out_chord + [self.sos_token_compound]
        else:   
            return metadata_tokens + out_chord + out

    def encode_series_labels_con_gen_commu(self, encoded_tokens):
        # meta_data_tokens,<SOC> chords, <EOC>, <SOS> music_seq, <EOS>   
        encoded_tokens = torch.tensor(encoded_tokens) #temporarily convert to tensor for easier slicing (len, 6)
        #first retrieve only the music_seq, retrieve the segment between [self.sos_token_compound] and [self.sos_token_compound]
        # Ensure the comparison results in a tensor
        sos_idx = (encoded_tokens == torch.tensor(self.sos_token_compound)).all(dim=1).nonzero(as_tuple=True)[0]
        eos_idx = (encoded_tokens == torch.tensor(self.eos_token_compound)).all(dim=1).nonzero(as_tuple=True)[0]
        encoded_tokens = encoded_tokens[sos_idx[0]+1:eos_idx[0]]

        #1. Convert onsets to delta onsets
        timeshift_labels_raw =  torch.diff(encoded_tokens[:, 0], prepend=torch.tensor([0]))
        #2. Concat the raw value

        output = torch.cat([torch.zeros(encoded_tokens.shape[0]).unsqueeze(-1) , timeshift_labels_raw.unsqueeze(-1), encoded_tokens[:, 1:]], dim = -1).tolist()

        #3. Add sos and eos label 
        output = [self.sos_label] + output + [self.eos_label]

        output = [self.sos_label for _ in range(sos_idx[0])]  + output #since we are not really predicting the chord and metadata tokens, we can use dummy labels (self.sos_label) for these tokens

        #4. encode it to labels 
        labels = [self.convert_to_language_tokens(x) for x in output]
        return labels


    def convert_to_language_tokens(self, x):
        """
        x looks like [time_shift, 1024, 11, 12, 129, 128]
        """
        out = []
        # for i in range(self.onset_vocab_size):
        #     out.append(self.onset_dict[x[i]]) #essentially this replicates onset bits
        out.append(self.sos_out_dict[x[0]])
        out.append(self.timeshift_dict[x[1]])
        out.append(self.duration_dict[x[2]])
        out.append(self.octave_dict[x[3]])  
        out.append(self.pitch_dict[x[4]])
        out.append(self.instrument_dict[x[5]])
        out.append(self.velocity_dict[x[6]])
        return out
    def convert_from_language_tokens(self, inp): 
        """
        x looks like [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1024, 11, 12, 129, 128]
        x looks like [time_shift, time_shift+1024, time_shift+1024+11, ...]
        """
        original_shape = inp.shape
        inp_flattened = inp.view(-1, original_shape[-1])
        out = []
        for x in inp_flattened: #(batch, vocab)
            timeshift = self.timeshift_dict_decode[x[0].item()]
            duration = self.duration_dict_decode[x[1].item()]
            octave = self.octave_dict_decode[x[2].item()]
            pitch = self.pitch_dict_decode[x[3].item()]
            # Pitch stays as integer cents (0-1199) when microtonal. The /100.0
            # conversion to fractional semitones happens in embed_tokens, which
            # handles both training (data loader cents) and inference (this path).
            instrument = self.instrument_dict_decode[x[4].item()]
            velocity = self.velocity_dict_decode[x[5].item()]
            out.append([timeshift, duration, octave, pitch, instrument, velocity])
        out = torch.tensor(out)
        out = out.view(*original_shape[:-1], -1)
        return out
    def add_new_tokens(self, token_name = "classification_token", token_val = -4):
        setattr(self, token_name, [token_val for _ in range(6)]) #example token
    def create_dur_dictionary(self):
        """Create a duration dictionary to map duration values to tokens in log scale."""
        dur_vocab_size_wo_soseos = self.dur_vocab_size - 2
        bin_width=10/dur_vocab_size_wo_soseos #assume duration ranges from 10ms to 10240ms 
        # Generate bin boundaries
        bin_boundaries = [2 ** (i * bin_width) for i in range(dur_vocab_size_wo_soseos + 1)]

        # Create dictionary to map tokens to bins
        token_to_bin = {}

        for token in range(1, 1025):  # Tokens from 1 to 1024
            # Determine which bin this token falls into
            for bin_num in range(dur_vocab_size_wo_soseos):
                if bin_boundaries[bin_num] <= token < bin_boundaries[bin_num + 1]:
                    token_to_bin[token] = bin_num + self.timeshift_vocab_size + self.sos_out_vocab_size

                    break

        # Handle the last token
        # token_to_bin[1024] = dur_vocab_size_wo_soseos - 1 
        token_to_bin[1024] = dur_vocab_size_wo_soseos - 1 + self.timeshift_vocab_size + self.sos_out_vocab_size

        #Add SOS and EOS dur 
        # print(f"sos_dur:{self.sos_dur + self.timeshift_vocab_size}, eos_dur:{self.eos_dur + self.timeshift_vocab_size}")
        token_to_bin[self.sos_dur] = self.dur_vocab_size - 2 + self.timeshift_vocab_size + self.sos_out_vocab_size #VERY UGLY FIX
        token_to_bin[self.eos_dur] = self.dur_vocab_size - 1 + self.timeshift_vocab_size + self.sos_out_vocab_size

        return token_to_bin

    @staticmethod
    def binary_to_decimal_batch(binary):
        """
        Converts a batch of binary representations to their decimal equivalents.

        Args:
            binary (torch.Tensor): A tensor containing binary numbers with each binary number in a row.

        Returns:
            torch.Tensor: A tensor containing the decimal equivalents of the input binary numbers.
        """
        # Create a mask to apply the binary weights
        bits = binary.size(-1) #2 bits are reserved for sos and eos TODO: add exception handling script, e.g., no eos and sos token is added, return dtype?
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(binary.device)

        # Apply the mask and sum up the results to get the decimal numbers
        dec = (binary[..., :] * mask).sum(dim=-1, keepdim=True)

        return dec
    
    @staticmethod
    def decimal_to_binary_batch(dec, bits):
        """
        Converts a batch of decimal numbers to their binary representations.

        Args:
            dec (torch.Tensor): A tensor containing decimal numbers.
            bits (int, optional): The desired number of bits in the binary representation. Default is 4.

        Returns:
            torch.Tensor: A tensor containing the binary representations of the input decimal numbers.
        """
        # Convert the input tensor to a long tensor for bitwise operations
        dec = dec.long()

        # Create a mask to extract the binary digits
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(dec.device)

        # Apply the mask to extract the binary digits
        binary = (dec.unsqueeze(-1) & mask).bool().to(torch.int8)

        return binary

    def midi_to_compound(self, midifile, TIME_RESOLUTION=100, debug=False, calibate_to_default_tempo=False):
        """
        modified from: https://github.com/jthickstun/anticipation/blob/main/anticipation/convert.py#L128
        midifile: a midi file path or a mido object
        output: a compound list, each item contains: [onset, duration, octave, pitch_class, instrument, velocity]

        When self.microtonal is True, pitchwheel messages are tracked per channel
        and used to compute a canonical pitch (float semitones) from which octave
        and pitch_class_cents (0-1199) are derived. Note-0 rest markers are filtered.
        """

        if type(midifile) == str:
            midi = mido.MidiFile(midifile)
        else:
            midi = midifile

        tokens = [] #output contains tuples of (onset, duration, octave, pitch_class, instr, velocity)
        note_idx = 0
        open_notes = defaultdict(list)
        filtered_note0 = []  # track filtered note-0 rest markers

        time = 0
        instruments = defaultdict(int) # default to code 0 = piano
        pitchbend_state = defaultdict(int)  # per-channel pitchbend (microtonal)
        tempo = 500000 # default tempo: 500000 microseconds per beat = 0.5 seconds
        for message in midi:
            time += message.time

            # sanity check: negative time?
            if message.time < 0:
                raise ValueError
            #messages: program_change, note_on, note_off
            if message.type == 'program_change':
                instruments[message.channel] = message.program #assign an instrument name to that channel
            elif message.type == 'pitchwheel':
                # Always track pitchbend state; only used when microtonal=True
                pitchbend_state[message.channel] = message.pitch
            elif message.type in ['note_on', 'note_off']:
                # special case: channel 9 is drums!
                instr = 128 if message.channel == 9 else instruments[message.channel]
                if message.type == 'note_on' and message.velocity > 0: # onset
                    # Filter out MIDI note 0: rest/silence marker (SymbTr convention)
                    if message.note == 0:
                        filtered_note0.append({
                            'time': time, 'time_ticks': round(TIME_RESOLUTION * time),
                            'channel': message.channel, 'velocity': message.velocity,
                            'pitchbend': pitchbend_state[message.channel], 'instrument': instr,
                        })
                        continue

                    # time quantization
                    if calibate_to_default_tempo:
                        time_in_ticks = round(TIME_RESOLUTION*time * 500000/tempo)
                    else:
                        time_in_ticks = round(TIME_RESOLUTION*time)

                    if self.microtonal:
                        # Compute canonical pitch from MIDI note + current pitchbend
                        bend_semitones = pitchbend_to_semitones(
                            pitchbend_state[message.channel], self.pitchbend_sensitivity)
                        canonical_pitch = message.note + bend_semitones
                        octave, pitch_class = pitch_to_octave_pitch_class_microtonal(canonical_pitch)
                    else:
                        octave, pitch_class = pitch_to_octave_pitch_class(message.note)

                    tokens.append([time_in_ticks, -1, octave, pitch_class, instr, message.velocity])
                    open_notes[(instr,message.note,message.channel)].append((note_idx, time))
                    note_idx += 1
                else: # offset
                    try:
                        open_idx, onset_time = open_notes[(instr,message.note,message.channel)].pop(0)
                    except IndexError:
                        if debug:
                            print('WARNING: ignoring bad offset')
                    else:
                        if calibate_to_default_tempo:
                            duration_ticks = round(TIME_RESOLUTION* 500000/tempo*(time-onset_time))
                        else:
                            duration_ticks = round(TIME_RESOLUTION*(time-onset_time))
                        if duration_ticks==0:
                            duration_ticks = 1
                        tokens[open_idx][1] = duration_ticks
            elif message.type == 'set_tempo':
                tempo = message.tempo
            elif message.type == 'time_signature':
                pass # we use real time
            elif message.type in ['aftertouch', 'polytouch', 'sequencer_specific']:
                pass # we don't attempt to model these
            elif message.type == 'control_change':
                pass # this includes pedal and per-track volume: ignore for now
            elif message.type in ['track_name', 'text', 'end_of_track', 'lyrics', 'key_signature',
                                'copyright', 'marker', 'instrument_name', 'cue_marker',
                                'device_name', 'sequence_number']:
                pass # possibly useful metadata but ignore for now
            elif message.type == 'channel_prefix':
                pass # relatively common, but can we ignore this?
            elif message.type in ['midi_port', 'smpte_offset', 'sysex']:
                pass
            else:
                if debug:
                    print('UNHANDLED MESSAGE', message.type, message)

        #assign a hard stop timing (dur = 250ms) for all open notes.
        unclosed_count = 0
        for (instr,note,channel) ,v in open_notes.items():
            unclosed_count += len(v)
            for (open_idx, onset_time) in v:
                tokens[open_idx][1] = TIME_RESOLUTION//4

        if debug and unclosed_count > 0:
            print(f'WARNING: {unclosed_count} unclosed notes')
            print('  ', midifile)

        # Log filtered note-0 rest markers
        if filtered_note0:
            import logging
            logger = logging.getLogger('MusicTokenizer')
            fname = midifile if isinstance(midifile, str) else '<mido.MidiFile object>'
            logger.warning(
                f'Filtered {len(filtered_note0)} note-0 rest marker(s) from {fname}. '
                f'Details: {filtered_note0}')
            if debug:
                print(f'NOTE-0 FILTER: {len(filtered_note0)} rest marker(s) removed from {fname}')
                for entry in filtered_note0:
                    print(f'  time={entry["time"]:.4f}s (tick={entry["time_ticks"]}), '
                          f'ch={entry["channel"]}, vel={entry["velocity"]}, '
                          f'pitchbend={entry["pitchbend"]}, instr={entry["instrument"]}')

        return tokens

    @staticmethod
    def compound_to_midi(tokens, TIME_RESOLUTION=100, debug=False,
                         microtonal=False, pitchbend_sensitivity=2.0):
        """Convert compound token array to a MIDI file.

        Args:
            tokens: array with shape (len, 6) — [onset, duration, octave, pitch, instrument, velocity].
                    In microtonal mode, pitch is integer cents (0-1199); otherwise integer pitch class (0-11).
            TIME_RESOLUTION: ticks per second (default 100).
            microtonal: if True, convert pitch from cents to semitones and emit pitchbend messages.
            pitchbend_sensitivity: pitchbend range in semitones (default 2.0, matching SymbTr encoding).
        """
        mid = mido.MidiFile()
        mid.ticks_per_beat = TIME_RESOLUTION // 2  # 2 beats/second at quarter=120

        time_index = defaultdict(list)

        for _, (row) in enumerate(tokens):
            time_in_ticks, duration, octave, pitch, instrument, velocity = row

            if microtonal:
                # Pitch is in cents (0-1199). Convert to fractional semitones then
                # decompose into MIDI note + pitchbend for sub-semitone accuracy
                pitch_frac = pitch / 100.0
                canonical = octave * 12 + pitch_frac
                note, pitchbend = canonical_pitch_to_midi_and_pitchbend(
                    canonical, pitchbend_sensitivity)
            else:
                note = octave_pitch_class_to_pitch(octave, int(pitch))
                pitchbend = 0

            time_index[(time_in_ticks, 0)].append((note, instrument, velocity, pitchbend))
            time_index[(time_in_ticks + duration, 1)].append((note, instrument, velocity, 0))

        track_idx = {}  # maps instrument to (track, current_time, channel_idx)
        num_tracks = 0
        for time_in_ticks, event_type in sorted(time_index.keys()):
            for (note, instrument, velocity, pb) in time_index[(time_in_ticks, event_type)]:
                if event_type == 0:  # onset
                    try:
                        track, previous_time, idx = track_idx[instrument]
                    except KeyError:
                        idx = num_tracks
                        previous_time = 0
                        track = mido.MidiTrack()
                        mid.tracks.append(track)
                        if instrument == 128:  # drums always go on channel 9
                            idx = 9
                            message = mido.Message('program_change', channel=idx, program=0)
                        else:
                            message = mido.Message('program_change', channel=idx, program=instrument)
                        track.append(message)
                        num_tracks += 1
                        if num_tracks == 9:
                            num_tracks += 1  # skip the drums track

                    # Insert pitchbend before note_on when sub-semitone offset is needed
                    if pb != 0:
                        track.append(mido.Message(
                            'pitchwheel', channel=idx, pitch=pb,
                            time=time_in_ticks - previous_time))
                        # Note_on follows immediately (delta time = 0)
                        track.append(mido.Message(
                            'note_on', note=note, channel=idx, velocity=velocity,
                            time=0))
                    else:
                        track.append(mido.Message(
                            'note_on', note=note, channel=idx, velocity=velocity,
                            time=time_in_ticks - previous_time))
                    track_idx[instrument] = (track, time_in_ticks, idx)
                else:  # offset
                    try:
                        track, previous_time, idx = track_idx[instrument]
                    except KeyError:
                        if debug:
                            print('IGNORING bad offset')
                        continue

                    track.append(mido.Message(
                        'note_off', note=note, channel=idx,
                        time=time_in_ticks - previous_time))
                    track_idx[instrument] = (track, time_in_ticks, idx)

        return mid

    def compound_to_midi_multi(self, list_of_compound_lists):
        out = []
        for _, compound_lists in enumerate(list_of_compound_lists):
            midi_out = self.compound_to_midi(
                compound_lists, microtonal=self.microtonal,
                pitchbend_sensitivity=self.pitchbend_sensitivity)
            out.append(midi_out)
        return out
