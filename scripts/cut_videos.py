import datetime
import os
import subprocess

def main():
    first_runtime = datetime.timedelta(minutes=2, seconds=9)
    second_runtime = datetime.timedelta(minutes=1, seconds=29)

    clip_length = 30

    for vid, runtime in zip(['cutFirstBiomass', 'cutSecondBiomass'], [first_runtime, second_runtime]):
        first_time = datetime.timedelta(minutes=0, seconds=0)
        clip = 1
        while first_time + datetime.timedelta(seconds=30) < first_runtime:
            command = f'ffmpeg -y -ss {first_time} -t 30 -i ../videos/{vid}.mkv ../videos/longer{clip}{vid}.mp4'
            subprocess.run(command, shell=True)
            first_time += datetime.timedelta(seconds=30)
            clip += 1

if __name__ == '__main__':
    main()