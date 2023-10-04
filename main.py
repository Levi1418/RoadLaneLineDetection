import cv2
import numpy as np
import matplotlib.pyplot as plt


def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def detect_lane_lines(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)

    height, width = edges.shape
    vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi / 180, threshold=40, minLineLength=50, maxLineGap=100)

    line_image = np.zeros_like(image)
    draw_lines(line_image, lines)

    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    return result


def draw_lines(image, lines, color=(255, 0, 0), thickness=5):
    if lines is None:
        return

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)


def process_image(image):
    result = detect_lane_lines(image)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detect_lane_lines(frame)
        cv2.imshow('Lane Lines', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    while True:
        media_type = input("Enter 'image' or 'video' to process (or 'exit' to quit): ")

        if media_type.lower() == 'image':
            image_path = input("Enter the path of the image file: ")
            image = cv2.imread(image_path)
            process_image(image)

        elif media_type.lower() == 'video':
            video_path = input("Enter the path of the video file: ")
            process_video(video_path)

        elif media_type.lower() == 'exit':
            break

        else:
            print("Invalid media type.")
            continue


if __name__ == '__main__':
    main()
