import base64
from pathlib import Path


def process_images():
    """
    Process all images in images/source folder and generate base64 strings.
    Saves each image as a separate .txt file in the images folder.
    """
    # Define paths
    source_dir = Path("images/source")
    output_dir = Path("images")

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    # Track count of processed images
    processed_count = 0

    # Valid image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']

    # Process each file in the source directory
    for file_path in source_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            try:
                # Read the image file
                with open(file_path, "rb") as img_file:
                    # Encode to base64
                    encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

                # Create output file path with original name but .txt extension
                image_name = file_path.stem
                output_file = output_dir / f"{image_name}.txt"

                # Save base64 string to text file
                with open(output_file, "w") as f:
                    f.write(encoded_string)

                processed_count += 1
                print(f"Processed: {file_path.name} -> {output_file.name}")
            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")

    print(f"Processed {processed_count} images")


if __name__ == "__main__":
    process_images()
