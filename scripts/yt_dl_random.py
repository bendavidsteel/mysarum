import random
import subprocess

playlists = [
    {
        'name': '80stvcommercials',
        'link': 'https://www.youtube.com/playlist?list=PLBF3772E61A5075CE',
        'num_videos': 1561
    }
]

def main():
    playlist = playlists[0]

    for _ in range(10):
        command = f"youtube-dl \
            --restrict-filenames \
            --playlist-items {random.randint(1, playlist['num_videos'])} \
            --extract-audio \
            {playlist['link']}"
        
        subprocess.run(command, shell=True)

if __name__ == '__main__':
    main()