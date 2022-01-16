#include <algorithm>
#include <vector>
#include <filesystem>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

constexpr const char *DEFAULT_INPUT_DIR = "./images/";
constexpr const char *DEFAULT_OUTPUT_DIR = "./results/";
constexpr const char *CLASSIFIER_FILE = "./haarcascade_russian_plate_number.xml";

/**
 * @brief Detect ROI with one license plate each by cv::CascadeClassifier
 * @param image Image with license plates
 * @param roi Regions of interest
 */
void detectROI(cv::Mat image, std::vector<cv::Rect> &roi)
{
    static cv::CascadeClassifier classifier(CLASSIFIER_FILE);
    // ROI is the whole image if CascadeClassifier cannot be loaded
    if (classifier.empty())
        roi = {cv::Rect(0, 0, image.cols, image.rows)};
    else
        classifier.detectMultiScale(image, roi, 1.2, 5);
}

/**
 * @brief Detect the contour of the license plate in image
 * @param image Image with one license plate
 * @return The biggest contour
 */
using Contour = std::vector<cv::Point>;
Contour dectContour(cv::Mat image)
{
    // Filter the blue area in the image
    cv::Mat diff(image.size(), CV_8UC1);
    auto itDiff = diff.begin<uchar>();
    for (auto itImage = image.begin<cv::Vec3b>(); itImage != image.end<cv::Vec3b>(); ++itImage, ++itDiff)
        *itDiff = static_cast<uchar>(std::max<short>((*itImage)[0] - ((*itImage)[1] >> 1) - ((*itImage)[2] >> 1), 0));

    // Connect the license plate regions through morphological operation
    cv::threshold(diff, diff, 128, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(6, 3));
    cv::morphologyEx(diff, diff, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(diff, diff, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(diff, diff, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(diff, diff, cv::MORPH_CLOSE, kernel);

    // Find and optimize all contours
    std::vector<Contour> contours;
    cv::findContours(diff, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    for (Contour &contour : contours)
    {
        Contour convex;
        cv::convexHull(contour, convex, true);
        cv::approxPolyDP(convex, contour, 3, true);
    }

    // Return the contour with biggest area
    contours.erase(std::remove_if(contours.begin(), contours.end(),
        [](const Contour &a) { return a.size() < 4; }), contours.end());
    if (contours.size())
    {
        return *std::max_element(contours.begin(), contours.end(),
            [](const Contour &a, const Contour &b) { return cv::contourArea(a) < contourArea(b); });
    }
    return Contour();
}

int main(void)
{
    namespace fs = std::filesystem;

    // Initialize folder interfaces
    const fs::path inputDir(DEFAULT_INPUT_DIR);
    const fs::path outputDir(DEFAULT_OUTPUT_DIR);
    const fs::directory_entry inputEntry(inputDir);
    if (inputEntry.status().type() != fs::file_type::directory)
        return 0; // Exit if inputDir not exists
    if (!fs::exists(outputDir))
        fs::create_directory(outputDir);

    // Travel inputDir non-recursively to find images
    for (const fs::directory_entry &imageEntry : fs::directory_iterator(inputEntry))
    {
        if (imageEntry.status().type() != fs::file_type::regular)
            continue;
        cv::Mat image = cv::imread(imageEntry.path().string());

        // Detect ROI with one license plate each
        std::vector<cv::Rect> roi;
        detectROI(image, roi);
        if (roi.size() == 0)
            continue;

        // Draw the contour of each license plate in ROI
        for (cv::Rect plate : roi)
        {
            cv::Mat plateImage = image(plate);
            Contour contour = dectContour(plateImage);
            if (contour.size())
                cv::polylines(plateImage, contour, true, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }

        // Save image to outputDir
        cv::imwrite((outputDir / imageEntry.path().filename()).string(), image);
    }
    return 0;
}
