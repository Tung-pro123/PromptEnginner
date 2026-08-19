import subprocess
import sys
import os

def convert_avi_to_mp4(input_file, output_file=None):
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
        
    if output_file is None:
        base, _ = os.path.splitext(input_file)
        output_file = base + '.mp4'

    print(f"Converting '{input_file}' to '{output_file}'...")
    
    # Command to run ffmpeg
    command = [
        'ffmpeg',
        '-i', input_file,
        '-c:v', 'libx264',      # Use H.264 video codec
        '-preset', 'medium',    # Encoding speed to compression ratio
        '-crf', '23',           # Constant Rate Factor (0-51, lower is better quality, 23 is default)
        '-c:a', 'aac',          # Use AAC audio codec
        '-b:a', '128k',         # Audio bitrate
        '-y',                   # Overwrite output file if it exists
        output_file
    ]
    
    try:
        # Run the command and capture output
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Conversion completed successfully! Saved as: '{output_file}'")
        return True
    except FileNotFoundError:
        print("Error: 'ffmpeg' is not installed or not found in the system path.")
        print("Please install FFmpeg (https://ffmpeg.org/download.html) and try again.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion:\n{e.stderr.decode('utf-8', errors='ignore')}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python avi_to_mp4.py <input_file.avi> [output_file.mp4]")
        print("Example: python avi_to_mp4.py video.avi")
        sys.exit(1)
        
    input_avi = sys.argv[1]
    output_mp4 = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_avi_to_mp4(input_avi, output_mp4)
