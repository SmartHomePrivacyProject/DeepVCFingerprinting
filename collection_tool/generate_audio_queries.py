import boto3
import pandas as pd
import config
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
        polly_client = boto3.Session(
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            region_name='eu-west-2').client('polly')
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

    else:
        f_name = 'voice_queries/amazon/google_voice_default/_' + q_file + "_.mp3"
        tts = gTTS(text=text_to_read, lang='en')
        tts.save(f_name)
print('Audio Files Generated and Saved to voice_queries/')
