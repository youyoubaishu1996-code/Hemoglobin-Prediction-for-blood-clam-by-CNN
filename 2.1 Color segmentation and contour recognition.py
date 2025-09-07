import cv2
import numpy as np


def process_red_regions(input_image):
    # Convert image from BGR to HSV color space
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

    # Define HSV range for red color
    lower_red1 = np.array([0, 50, 30])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 30])
    upper_red2 = np.array([180, 255, 255])

    # Create mask for red regions
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Set non-red regions to white
    result = input_image.copy()
    result[red_mask == 0] = [255, 255, 255]  # BGR white

    return result


def detect_and_save_contours(input_path, output_path):
    # Read input image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Unable to read image {input_path}")
        return

    # Process red regions first
    processed_image = process_red_regions(image)

    # Convert to HSV and create mask for red regions again
    hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 30])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 30])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Merge overlapping contours
    merged_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = (x, y, x + w, y + h)
        overlapped = False

        for i, m_roi in enumerate(merged_contours):
            if not (roi[2] < m_roi[0] or roi[0] > m_roi[2] or roi[3] < m_roi[1] or roi[1] > m_roi[3]):
                # Merge overlapping regions
                x_min = min(x, m_roi[0])
                y_min = min(y, m_roi[1])
                x_max = max(x + w, m_roi[2])
                y_max = max(y + h, m_roi[3])
                merged_contours[i] = (x_min, y_min, x_max, y_max)
                overlapped = True
                break

        if not overlapped:
            merged_contours.append(roi)

    # Calculate contour areas and sort
    areas = [(i, (rect[2] - rect[0]) * (rect[3] - rect[1])) for i, rect in enumerate(merged_contours)]
    areas.sort(key=lambda x: x[1])  # Sort by area ascending

    print("All sorted areas:", [x[1] for x in areas])

    # Filter out small contours (area < 10)
    filtered_areas = [area for area in areas if area[1] >= 10]
    removed_areas = [area for area in areas if area[1] < 10]

    # Create result image
    result_image = processed_image.copy()

    # Set removed contours to white
    for index, _ in removed_areas:
        x, y, x2, y2 = merged_contours[index]
        result_image[y:y2, x:x2] = [255, 255, 255]  # Fill with white

    # Draw remaining contours
    print(f"Found {len(filtered_areas)} contours")
    for index, _ in filtered_areas:
        x, y, x2, y2 = merged_contours[index]

        # Calculate width and height
        width = x2 - x
        height = y2 - y

        # Calculate center point
        center_x = x + width / 2
        center_y = y + height / 2

        # Expand dimensions by 1.2 times
        new_width = width * 1
        new_height = height * 1

        # Calculate new coordinates
        new_x = int(center_x - new_width / 2)
        new_y = int(center_y - new_height / 2)
        new_x2 = int(center_x + new_width / 2)
        new_y2 = int(center_y + new_height / 2)

        # Ensure coordinates are within image bounds
        height, width = result_image.shape[:2]
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_x2 = min(width - 1, new_x2)
        new_y2 = min(height - 1, new_y2)

        print(new_x, new_y, new_x2, new_y2)
        cv2.rectangle(result_image, (new_x, new_y), (new_x2, new_y2), (0, 0, 0), 2)

    # Save result
    cv2.imwrite(output_path, result_image)
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    # First process the original image to extract red regions
    input_image = cv2.imread('1.png')
    if input_image is None:
        print("Error: Unable to read input image '1.png'")
    else:
        red_processed = process_red_regions(input_image)

        # Then detect contours and save
        detect_and_save_contours('1.png', 'contours_output.png')