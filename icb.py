import cv2
import numpy as np
import time

# Initialize camera
cap = cv2.VideoCapture(0)  # Adjust '0' to the correct camera index

# Game specific variables
total_score = 0
current_hole_value = 100  # Adjust based on game specifics
num_balls = 3
completed_loops = 0
current_ball = 1
game_over = False

# Timing and scoring
start_time = time.time()
time_per_hole = []
loop_start_time = None
fastest_loop_time = None

# Define hole positions and characteristics here (you will need to customize this)
hole_positions = []  # Fill with the coordinates or identifiers for each hole

# Function to initialize the game environment (put setup tasks here)
def initialize_game():
    global total_score, num_balls, completed_loops, current_ball, game_over
    global time_per_hole, loop_start_time, fastest_loop_time
    # Reset or setup game variables
    total_score = 0
    num_balls = 3
    completed_loops = 0
    current_ball = 1
    game_over = False
    time_per_hole = []
    loop_start_time = time.time()
    fastest_loop_time = None
    # Add more initialization as needed

initialize_game()

while not game_over:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break  # If no frame is captured, break the loop

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    ret, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # Convert to HSV color space for color filtering
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_silver = np.array([0, 0, 100])  # adjust these values
    upper_silver = np.array([180, 40, 255])  # adjust these values
    color_mask = cv2.inRange(hsv, lower_silver, upper_silver)
    
    # Combine masks from grayscale threshold and color filtering for robust detection
    combined_mask = cv2.bitwise_and(thresh, color_mask)

    # Use morphological operations to clean up the mask
    combined_mask = cv2.erode(combined_mask, None, iterations=2)
    combined_mask = cv2.dilate(combined_mask, None, iterations=2)

    # Find contours in the combined mask
    contours, hierarchy = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detecting the ball
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        if 10 < radius < 30:  # Adjust radius values based on the actual size of the ball
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

    # Detecting the lit hole
    lit_threshold = 250  # for grayscale, close to 255 means very bright
    for hole in hole_positions:
        # Assuming hole positions are pre-defined with x, y, width, and height
        hole_roi = gray[hole['y']:hole['y']+hole['height'], hole['x']:hole['x']+hole['width']]
        if cv2.mean(hole_roi)[0] > lit_threshold:
            cv2.rectangle(frame, (hole['x'], hole['y']), (hole['x'] + hole['width'], hole['y'] + hole['height']), (0, 255, 0), 2)

    # Update game state
    for hole in hole_positions:
        distance = np.sqrt((x - hole['center_x'])**2 + (y - hole['center_y'])**2)
        if distance < radius + hole['radius']:  # Adjust this condition based on your game
            if hole['is_lit']:
                total_score += current_hole_value
                # Update for next hole or loop
            else:
                num_balls -= 1
                if num_balls == 0:
                    game_over = True  # End the game if out of balls
            break  # Stop checking other holes

    # Display the frame for debugging
    cv2.imshow('Game', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, rçelease the capture and close windows
cap.release()
cv2.destroyAllWindows()


import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture(0)

def calculate_likelihood(contour, expected_radius):
    # Initialize scores
    size_score = 0
    circularity_score = 0
    
    # Calculate contour properties
    ((x, y), radius) = cv2.minEnclosingCircle(contour)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Size score based on how close the radius is to the expected radius
    if 10 < radius < 30:  # Assuming the expected radius range
        size_score = 1 - (abs(radius - expected_radius) / expected_radius)
    
    # Circularity score
    if area > 0 and perimeter > 0:
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        # Assuming a perfect circle has a circularity of 1
        circularity_score = max(0, 1 - abs(1 - circularity))
    
    # Combine scores for an overall likelihood (you can weigh these if one is more important)
    overall_likelihood = (size_score + circularity_score) / 2
    return overall_likelihood * 100  # Convert to percentage

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Morphological operations to reduce small noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        likelihood = calculate_likelihood(contour, expected_radius=20)  # Adjust expected_radius as needed
        if likelihood > 50:  # Adjust the threshold as needed
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            # Display the likelihood on the frame
            cv2.putText(frame, f"{likelihood:.2f}%", (int(x - radius), int(y - radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    # Display the original frame with detected ball
    cv2.imshow('Ball Detection', frame)
    cv2.imshow('Thresholded', thresh)  # For debugging

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
