#!/usr/bin/env python3

import boto3
import pandas as pd
import argparse
from gtts import gTTS

"""
CLI Script that generates audio files for a list of queries.  Can generate 5 different text to speach voicese
from Google and AWS services. Be sure to add AWS credentials if you plan to use the AWS Polly voices.
"""

# Get args from CLI
parser = argparse.ArgumentParser()
parser.add_argument("csv", help='A CSV file with queries with a column labeled Query')
parser.add_argument('--wake_word', help='The wake word that initiates an interaction with the device, default is '
                                        'Alexa')
parser.add_argument('--voice', help='Specify which voice to use.  Defaults to the default US english google '
                                    'female voice.  optional voices include joanna, salli, matt or joey')

args = parser.parse_args()
csv_path = args.csv

# Five possible voices
joanna = False
salli = False
matt = False
joey = False
polly = False
kendra = False
ivy = False
kimberly = False
justin = False
kevin = False
russell = False
amy = False
emma = False
brian = False
nicole = False
olivia = False
aditi = False
raveena = False

# Set flags if using the Polly voices
if args.voice == 'joanna':
    print('Using Polly Joanna Voice')
    joanna = True
    polly = True
elif args.voice == 'salli':
    print('Using Polly Salli Voice')
    salli = True
    polly = True
elif args.voice == 'matt':
    print('Using Polly Matthew Voice')
    matt = True
    polly = True
elif args.voice == 'joey':
    print('Using Polly Joey Voice')
    joey = True
    polly = True
elif args.voice == 'kendra':
    print('Using Polly Kendra Voice')
    kendra = True
    polly = True
elif args.voice == 'ivy':
    print('Using Polly Ivy Voice')
    ivy = True
    polly = True
elif args.voice == 'kimberly':
    print('Using Polly Kimberly Voice')
    kimberly = True
    polly = True
elif args.voice == 'justin':
    print('Using Polly Justin Voice')
    justin = True
    polly = True
elif args.voice == 'kevin':
    print('Using Polly Kevin Voice')
    kevin = True
    polly = True
elif args.voice == 'russell':
    print('Using Polly Russell Voice')
    russell = True
    polly = True
elif args.voice == 'amy':
    print('Using Polly Amy Voice')
    amy = True
    polly = True
elif args.voice == 'emma':
    print('Using Polly Emma Voice')
    emma = True
    polly = True
elif args.voice == 'brian':
    print('Using Polly Brian Voice')
    brian = True
    polly = True
elif args.voice == 'nicole':
    print('Using Polly Nicole Voice')
    nicole = True
    polly = True
elif args.voice == 'olivia':
    print('Using Polly Olivia Voice')
    olivia = True
    polly = True
elif args.voice == 'aditi':
    print('Using Polly Aditi Voice')
    aditi = True
    polly = True
elif args.voice == 'raveena':
    print('Using Polly Raveena Voice')
    raveena = True
    polly = True

# Set default wakeword as Alexa, i.e. Amazon Echo Devices
default_wake_word = "Alexa "
if args.wake_word is not None:
    default_wake_word = args.wake_word + " "
    
# Read in the query list from a csv file containing a list of queries
queries = pd.read_csv(csv_path)

# Iterate through the query list and generate the correct mp3 audio files cor
for q in queries['Query']:
    text_to_read = default_wake_word + '  ,   ' + q
    q_file = q.replace(" ", "_")
    if polly:
        polly_client = boto3.client('polly')
        if joanna:
            f_name = 'voice_queries/amazon/polly/joanna/_' + q_file + "_.mp3"
            response = polly_client.synthesize_speech(VoiceId='Joanna',
                                                      OutputFormat='mp3',
                                                      Text=text_to_read + "              ")
            file = open(f_name, 'wb')
            file.write(response['AudioStream'].read())
            file.close()

        elif salli:
            f_name = 'voice_queries/amazon/polly/salli/_' + q_file + "_.mp3"
            response = polly_client.synthesize_speech(VoiceId='Salli',
                                                      OutputFormat='mp3',
                                                      Text=text_to_read + "              ")
            file = open(f_name, 'wb')
            file.write(response['AudioStream'].read())
            file.close()

        elif matt:
            f_name = 'voice_queries/amazon/polly/matt/_' + q_file + "_.mp3"
            response = polly_client.synthesize_speech(VoiceId='Matthew',
                                                      OutputFormat='mp3',
                                                      Text=text_to_read + "              ")
            file = open(f_name, 'wb')
            file.write(response['AudioStream'].read())
            file.close()

        elif joey:
            f_name = 'voice_queries/amazon/polly/joey/_' + q_file + "_.mp3"
            response = polly_client.synthesize_speech(VoiceId='Joey',
                                                      OutputFormat='mp3',
                                                      Text=text_to_read + "              ")
            file = open(f_name, 'wb')
            file.write(response['AudioStream'].read())
            file.close()

        elif kendra:
            f_name = 'voice_queries/amazon/polly/kendra/_' + q_file + "_.mp3"
            response = polly_client.synthesize_speech(VoiceId='Kendra',
                                                      OutputFormat='mp3',
                                                      Text=text_to_read + "              ")
            file = open(f_name, 'wb')
            file.write(response['AudioStream'].read())
            file.close()

        elif ivy:
            f_name = 'voice_queries/amazon/polly/ivy/_' + q_file + "_.mp3"
            response = polly_client.synthesize_speech(VoiceId='Ivy',
                                                      OutputFormat='mp3',
                                                      Text=text_to_read + "              ")
            file = open(f_name, 'wb')
            file.write(response['AudioStream'].read())
            file.close()

        elif kimberly:
            f_name = 'voice_queries/amazon/polly/kimberly/_' + q_file + "_.mp3"
            response = polly_client.synthesize_speech(VoiceId='Kimberly',
                                                      OutputFormat='mp3',
                                                      Text=text_to_read + "              ")
            file = open(f_name, 'wb')
            file.write(response['AudioStream'].read())
            file.close()
        
        elif justin:
            f_name = 'voice_queries/amazon/polly/justin/_' + q_file + "_.mp3"
            response = polly_client.synthesize_speech(VoiceId='Justin',
                                                      OutputFormat='mp3',
                                                      Text=text_to_read + "              ")
            file = open(f_name, 'wb')
            file.write(response['AudioStream'].read())
            file.close()

        elif kevin:
            f_name = 'voice_queries/amazon/polly/kevin/_' + q_file + "_.mp3"
            response = polly_client.synthesize_speech(VoiceId='Kevin',
                                                      Engine="neural",
                                                      OutputFormat='mp3',
                                                      Text=text_to_read + "              ")
            file = open(f_name, 'wb')
            file.write(response['AudioStream'].read())
            file.close()
        elif russell:
            f_name = 'voice_queries/amazon/polly/russell/_' + q_file + "_.mp3"
            response = polly_client.synthesize_speech(VoiceId='Russell',
                                                      OutputFormat='mp3',
                                                      Text=text_to_read + "              ")
            file = open(f_name, 'wb')
            file.write(response['AudioStream'].read())
            file.close()
        elif amy:
            f_name = 'voice_queries/amazon/polly/amy/_' + q_file + "_.mp3"
            response = polly_client.synthesize_speech(VoiceId='Amy',
                                                      OutputFormat='mp3',
                                                      Text=text_to_read + "              ")
            file = open(f_name, 'wb')
            file.write(response['AudioStream'].read())
            file.close()
        elif emma:
            f_name = 'voice_queries/amazon/polly/emma/_' + q_file + "_.mp3"
            response = polly_client.synthesize_speech(VoiceId='Emma',
                                                      OutputFormat='mp3',
                                                      Text=text_to_read + "              ")
            file = open(f_name, 'wb')
            file.write(response['AudioStream'].read())
            file.close()
        elif brian:
            f_name = 'voice_queries/amazon/polly/brian/_' + q_file + "_.mp3"
            response = polly_client.synthesize_speech(VoiceId='Brian',
                                                      OutputFormat='mp3',
                                                      Text=text_to_read + "              ")
            file = open(f_name, 'wb')
            file.write(response['AudioStream'].read())
            file.close()
        elif nicole:
            f_name = 'voice_queries/amazon/polly/nicole/_' + q_file + "_.mp3"
            response = polly_client.synthesize_speech(VoiceId='Nicole',
                                                      OutputFormat='mp3',
                                                      Text=text_to_read + "              ")
            file = open(f_name, 'wb')
            file.write(response['AudioStream'].read())
            file.close()
        elif olivia:
            f_name = 'voice_queries/amazon/polly/olivia/_' + q_file + "_.mp3"
            response = polly_client.synthesize_speech(VoiceId='Olivia',
                                                      Engine="neural",  
                                                      OutputFormat='mp3',
                                                      Text=text_to_read + "              ")
            file = open(f_name, 'wb')
            file.write(response['AudioStream'].read())
            file.close()
        elif aditi:
            f_name = 'voice_queries/amazon/polly/aditi/_' + q_file + "_.mp3"
            response = polly_client.synthesize_speech(VoiceId='Aditi',
                                                      OutputFormat='mp3',
                                                      Text=text_to_read + "              ")
            file = open(f_name, 'wb')
            file.write(response['AudioStream'].read())
            file.close()
        elif raveena:
            f_name = 'voice_queries/amazon/polly/raveena/_' + q_file + "_.mp3"
            response = polly_client.synthesize_speech(VoiceId='Raveena',
                                                      OutputFormat='mp3',
                                                      Text=text_to_read + "              ")
            file = open(f_name, 'wb')
            file.write(response['AudioStream'].read())
            file.close()
        else:
            print("Invalid name.")

    else:
        f_name = 'voice_queries/amazon/google_voice_default/_' + q_file + "_.mp3"
        tts = gTTS(text=text_to_read, lang='en')
        tts.save(f_name)
print('Audio Files Generated and Saved to voice_queries/')
