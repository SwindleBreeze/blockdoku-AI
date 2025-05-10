import json
import os
import glob

def join_json_game_files(input_directory: str, output_filename: str):
    """
    Finds all JSON files in the input_directory, assumes they contain a "games" list,
    and combines all these "games" lists into a single new JSON file.

    Args:
        input_directory (str): The path to the directory containing the JSON files.
        output_filename (str): The name of the output JSON file to create.
    """
    all_combined_games = []
    
    # Use glob to find all .json files in the directory
    json_files = glob.glob(os.path.join(input_directory, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in directory: {input_directory}")
        return

    print(f"Found {len(json_files)} JSON files to process: {json_files}")

    for file_path in json_files:
        # Skip the output file itself if it's in the same directory and already exists
        if os.path.basename(file_path) == os.path.basename(output_filename) and os.path.dirname(file_path) == input_directory :
            print(f"Skipping potential output file: {file_path}")
            continue
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and "games" in data and isinstance(data["games"], list):
                    all_combined_games.extend(data["games"])
                    print(f"Successfully processed and added {len(data['games'])} games from: {file_path}")
                else:
                    print(f"Warning: File {file_path} does not have the expected format (missing 'games' list). Skipping.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from file {file_path}. Skipping.")
        except Exception as e:
            print(f"Warning: An error occurred while processing file {file_path}: {e}. Skipping.")

    if not all_combined_games:
        print("No game data was successfully combined.")
        return

    output_data = {"games": all_combined_games}
    output_path = os.path.join(input_directory, output_filename) # Save in the same directory by default

    try:
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2) # Using indent for readability
        print(f"\nSuccessfully combined {len(all_combined_games)} games into: {output_path}")
    except IOError:
        print(f"Error: Could not write to output file: {output_path}")
    except Exception as e:
        print(f"An unexpected error occurred while writing output file: {e}")

if __name__ == '__main__':
    input_dir_original = "human_games"
    output_file_original = "combined_human_games_original.json"
    print(f"\n--- Joining original format JSONs from '{input_dir_original}' ---")
    join_json_game_files(input_dir_original, output_file_original)