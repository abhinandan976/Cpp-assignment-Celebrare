#include <iostream>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

torch::Tensor tensorizeImage(cv::Mat img, int size) {
    cv::resize(img, img, cv::Size(size, size));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32F, 1.0 / 255.0);
    
    auto tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3});
    tensor = tensor.permute({0, 3, 1, 2});
    tensor = tensor.sub(0.5).mul(2.0);
    
    return tensor;
}

cv::Mat unTensorizeImage(torch::Tensor tensor) {
    tensor = tensor.add(1.0).div(2.0).clamp(0.0, 1.0);
    tensor = tensor.permute({0, 2, 3, 1}).contiguous();
    tensor = tensor.squeeze(0);
    tensor = tensor.mul(255).to(torch::kU8);
    
    cv::Mat result(tensor.size(0), tensor.size(1), CV_8UC3, tensor.data_ptr<uchar>().clone());
    cv::cvtColor(result, result, cv::COLOR_RGB2BGR);
    
    return result;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./enhancer <input_image> <output_image>\n";
        return 1;
    }

    std::string inputPath = argv[1];
    std::string outputPath = argv[2];
    
    const std::string gfpganPath = "gfpgan_v1.4_traced.pt";
    const std::string faceProtoPath = "deploy.prototxt";
    const std::string faceModelPath = "res10_300x300_ssd_iter_140000.caffemodel";
    const int modelSize = 512;

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Executing on device: " << device << std::endl;

    torch::jit::script::Module module;
    cv::dnn::Net net;

    try {
        module = torch::jit::load(gfpganPath);
        module.to(device);
        module.eval();
        net = cv::dnn::readNetFromCaffe(faceProtoPath, faceModelPath);
    } catch (const std::exception& e) {
        std::cerr << "Error loading models: " << e.what() << std::endl;
        return 1;
    }

    cv::Mat image = cv::imread(inputPath);
    if (image.empty()) {
        std::cerr << "Failed to load input image.\n";
        return 1;
    }
    cv::Mat finalImage = image.clone();

    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(300, 300), {104.0, 177.0, 123.0});
    net.setInput(blob);
    cv::Mat detections = net.forward();
    
    cv::Mat detectionMatrix(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());

    for (int i = 0; i < detectionMatrix.rows; ++i) {
        float confidence = detectionMatrix.at<float>(i, 2);

        if (confidence > 0.75) {
            std::cout << "Processing detected face with confidence: " << confidence << std::endl;
            
            int x1 = static_cast<int>(detectionMatrix.at<float>(i, 3) * image.cols);
            int y1 = static_cast<int>(detectionMatrix.at<float>(i, 4) * image.rows);
            int x2 = static_cast<int>(detectionMatrix.at<float>(i, 5) * image.cols);
            int y2 = static_cast<int>(detectionMatrix.at<float>(i, 6) * image.rows);

            cv::Rect faceBox(x1, y1, x2 - x1, y2 - y1);
            cv::Mat face = image(faceBox);

            if (face.empty()) continue;

            torch::Tensor inputTensor = tensorizeImage(face.clone(), modelSize).to(device);
            
            torch::Tensor outputTensor = module.forward({inputTensor}).toTensor();
            
            cv::Mat enhancedFace = unTensorizeImage(outputTensor.to(torch::kCPU));
            
            cv::resize(enhancedFace, enhancedFace, faceBox.size());
            enhancedFace.copyTo(finalImage(faceBox));
            
            break; 
        }
    }

    if (cv::imwrite(outputPath, finalImage)) {
        std::cout << "Enhanced image saved to " << outputPath << std::endl;
    } else {
        std::cerr << "Failed to save the output image.\n";
        return 1;
    }

    return 0;
}