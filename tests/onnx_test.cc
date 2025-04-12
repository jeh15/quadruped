#include <filesystem>
#include <iostream>

#include "rules_cc/cc/runfiles/runfiles.h"

#include <onnxruntime_cxx_api.h>


using rules_cc::cc::runfiles::Runfiles;


int main(int argc, char** argv) {
    // Use Runfiles to get the path to the model:
    std::string error;
    std::unique_ptr<Runfiles> runfiles(
        Runfiles::Create(argv[0], BAZEL_CURRENT_REPOSITORY, &error)
    );

    std::filesystem::path model_path = 
        runfiles->Rlocation("unitree-interface/onnx_models/breezy-dawn-149.onnx");

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeTest");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::Session session = Ort::Session(env, model_path.c_str(), session_options);
    
    // Initialize Inputs and Outputs:
    // Ort::AllocatorWithDefaultOptions allocator;
    // std::vector<std::string> input_names;
    // std::vector<std::string> output_names;
    // std::vector<std::vector<int64_t>> input_shapes;
    // std::vector<std::vector<int64_t>> output_shapes;

    // // Get Input Names and Shapes
    // for (size_t i = 0; i < session.GetInputCount(); ++i) {
    //     input_names.push_back(session.GetInputName(i));
    //     Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    //     auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    //     input_shapes.push_back(tensor_info.GetShape());
    // }

    // // Get Output Names and Shapes
    // for (size_t i = 0; i < session.GetOutputCount(); ++i) {
    //     output_names.push_back(session.GetOutputName(i));
    //     Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
    //     auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    //     output_shapes.push_back(tensor_info.GetShape());
    // }

    // // Print Input and Output Names and Shapes
    // std::cout << "Input Names: " << std::endl;
    // for (const auto& name : input_names) {
    //     std::cout << name << std::endl;
    // }
    // std::cout << "Input Shapes: " << std::endl;
    // for (const auto& shape : input_shapes) {
    //     std::cout << "[";
    //     for (const auto& dim : shape) {
    //         std::cout << dim << " ";
    //     }
    //     std::cout << "]" << std::endl;
    // }
    // std::cout << "Output Names: " << std::endl;
    // for (const auto& name : output_names) {
    //     std::cout << name << std::endl;
    // }
    // std::cout << "Output Shapes: " << std::endl;
    // for (const auto& shape : output_shapes) {
    //     std::cout << "[";
    //     for (const auto& dim : shape) {
    //         std::cout << dim << " ";
    //     }
    //     std::cout << "]" << std::endl;
    // }

    return 0;
}