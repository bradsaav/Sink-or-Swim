#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>  // For sqrt and pow
#include <algorithm>  // For transform
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>  // For timing


using namespace std;
using namespace chrono;

struct Instance {
    int class_label;
    vector<double> features;
};

int classify(const vector<Instance>& training_data, const Instance& test_instance, const vector<int>& feature_subset) {
    double min_distance = numeric_limits<double>::max();
    int predicted_class = -1;

    for (const auto& train_instance : training_data) {
        double distance = 0.0;
        for (int feature : feature_subset) {
            distance += pow(test_instance.features[feature - 1] - train_instance.features[feature - 1], 2);
        }
        distance = sqrt(distance);

        if (distance < min_distance) {
            min_distance = distance;
            predicted_class = train_instance.class_label;
        }
    }
    return predicted_class;
}

double leave_one_out_validation(const vector<Instance>& dataset, const vector<int>& feature_subset) {
     int correct_predictions = 0;

    for (size_t i = 0; i < dataset.size(); ++i) {
        vector<Instance> training_data;
        for (size_t j = 0; j < dataset.size(); ++j) {
            if (i != j) {
                training_data.push_back(dataset[j]);
            }
        }

        const Instance& test_instance = dataset[i];
        int predicted_class = classify(training_data, test_instance, feature_subset);
        if (predicted_class == test_instance.class_label) {
            ++correct_predictions;
        }
    }

    return (correct_predictions * 100.0) / dataset.size();  // Return accuracy in percentage
}

vector<Instance> load_data(const string& file_name) {
    vector<Instance> dataset;
    ifstream file(file_name);
    if (!file) {
        cerr << "Error opening file: " << file_name << endl;
        exit(EXIT_FAILURE);
    }

    string line;
    while (getline(file, line)) {
        // Skip empty lines
        if (line.empty()) {
            continue;
        }

        istringstream iss(line);
        Instance instance;
        iss >> instance.class_label;

        double value;
        vector<double> features;
        while (iss >> value) {
            // Skip the first feature if it's always 0
            if (features.empty() && value == 0) {
                continue;
            }
            features.push_back(value);
        }

        // Ensure that instances with at least one feature are added
        if (!features.empty()) {
            instance.features = features;
            dataset.push_back(instance);
        } else {
            cerr << "Warning: Instance with no features found. Skipping." << endl;
        }
    }

     // Normalize features using z-score normalization
    if (!dataset.empty()) {
        size_t num_features = dataset[0].features.size();
        vector<double> means(num_features, 0.0);
        vector<double> std_devs(num_features, 0.0);

        // Calculate mean for each feature
        for (const auto& instance : dataset) {
            for (size_t i = 0; i < num_features; ++i) {
                means[i] += instance.features[i];
            }
        }
        for (size_t i = 0; i < num_features; ++i) {
            means[i] /= dataset.size();
        }

        // Calculate standard deviation for each feature
        for (const auto& instance : dataset) {
            for (size_t i = 0; i < num_features; ++i) {
                std_devs[i] += pow(instance.features[i] - means[i], 2);
            }
        }
        for (size_t i = 0; i < num_features; ++i) {
            std_devs[i] = sqrt(std_devs[i] / dataset.size());
        }

        // Normalize each feature value using z-score normalization
        for (auto& instance : dataset) {
            for (size_t i = 0; i < num_features; ++i) {
                if (std_devs[i] != 0) {  // Avoid division by zero
                    instance.features[i] = (instance.features[i] - means[i]) / std_devs[i];
                } else {
                    instance.features[i] = 0.0;  // If all values are the same, normalize to 0
                }
            }
        }
    }


    return dataset;
}

// Forward Selection
void forward_selection(int num_features, const vector<Instance>& dataset) {
    cout << "Running Forward Selection Algorithm..." << endl;

    vector<int> best_features;
    double best_accuracy = 0.0;

    // Variables to track the best overall feature subset
    vector<int> best_overall_features;
    double best_overall_accuracy = 0.0;

     auto start_time = high_resolution_clock::now();

    cout << "\nStarting search with no features selected." << endl;
    for (int i = 0; i < num_features; ++i) {
        double current_best_accuracy = 0.0;
        int current_best_feature = -1;

        for (int feature = 1; feature <= num_features; ++feature) {
            if (find(best_features.begin(), best_features.end(), feature) == best_features.end()) {
                vector<int> current_features = best_features;
                current_features.push_back(feature);
                double accuracy = leave_one_out_validation(dataset, current_features);
                cout << "Using features { ";
                for (int f : current_features) cout << f << " ";
                cout << "} accuracy is " << accuracy << "%" << endl;

                if (accuracy > current_best_accuracy) {
                    current_best_accuracy = accuracy;
                    current_best_feature = feature;
                }
            }
        }

        if (current_best_feature != -1) {
            best_features.push_back(current_best_feature);
            best_accuracy = current_best_accuracy;

            // Update the overall best feature set and accuracy
            if (best_accuracy > best_overall_accuracy) {
                best_overall_accuracy = best_accuracy;
                best_overall_features = best_features;
            }

            cout << "Best feature set so far: { ";
            for (int f : best_features) cout << f << " ";
            cout << "} with accuracy " << best_accuracy << "%" << endl << endl;
        }
    }
    auto end_time = high_resolution_clock::now();
    auto duration_temp = end_time - start_time;
    duration<double> duration = duration_temp;
    // Print the best overall feature set and accuracy
    cout << "Finished search!! The best feature subset is { ";
    for (int f : best_overall_features) cout << f << " ";
    cout << "} with an accuracy of " << best_overall_accuracy << "%" << endl;
    cout << "Execution Time: " << duration.count() << " seconds." << endl;
}

// Backward Elimination
void backward_elimination(int num_features, const vector<Instance>& dataset) {
    cout << "Running Backward Elimination Algorithm..." << endl;

    vector<int> current_features(num_features);
    for (int i = 0; i < num_features; ++i) current_features[i] = i + 1;

    double best_accuracy = leave_one_out_validation(dataset, current_features);
    vector<int> best_features = current_features;

    vector<int> best_overall_features = current_features;
    double best_overall_accuracy = best_accuracy;

    auto start_time = high_resolution_clock::now();

    cout << "\nStarting search with all features selected: { ";
    for (int f : current_features) cout << f << " ";
    cout << "} Initial accuracy: " << best_accuracy << "%" << endl;

    for (int i = 0; i < num_features - 1; ++i) {
        double current_best_accuracy = 0.0;
        int feature_to_remove = -1;

        for (int feature : current_features) {
            vector<int> temp_features = current_features;
            temp_features.erase(remove(temp_features.begin(), temp_features.end(), feature), temp_features.end());

            double accuracy = leave_one_out_validation(dataset, temp_features);
            cout << "Using features { ";
            for (int f : temp_features) cout << f << " ";
            cout << "} accuracy is " << accuracy << "%" << endl;

            if (accuracy > current_best_accuracy) {
                current_best_accuracy = accuracy;
                feature_to_remove = feature;
            }
        }

        if (feature_to_remove != -1) {
            current_features.erase(remove(current_features.begin(), current_features.end(), feature_to_remove), current_features.end());
            best_accuracy = current_best_accuracy;
            best_features = current_features;

            if (best_accuracy > best_overall_accuracy) {
                best_overall_accuracy = best_accuracy;
                best_overall_features = best_features;
            }


            cout << "Best feature set so far: { ";
            for (int f : best_features) cout << f << " ";
            cout << "} with accuracy " << best_accuracy << "%" << endl << endl;
        }
    }

    auto end_time = high_resolution_clock::now();
    auto duration_temp = end_time - start_time;
    duration<double> duration = duration_temp;
    cout << "Finished search!! The best feature subset is { ";
    for (int f : best_overall_features) cout << f << " ";
    cout << "} with an accuracy of " << best_overall_accuracy << "%" << endl;
    cout << "Execution Time: " << duration.count() << " seconds." << endl;
}

int main() {
    string file_name;
    cout << "Enter the dataset file name: ";
    cin >> file_name;

    vector<Instance> dataset = load_data(file_name);
    int num_features = dataset[0].features.size();
    cout << "Dataset loaded with " << num_features << " features and " << dataset.size() << " instances." << endl;

    int choice;
    cout << "Choose an algorithm to run:\n1. Forward Selection\n2. Backward Elimination\n";
    cin >> choice;

    if (choice == 1) {
        forward_selection(num_features, dataset);
    } else if (choice == 2) {
        backward_elimination(num_features, dataset);
    } else {
        cout << "Invalid choice!" << endl;
    }

    vector<int> test_feature_one = {3, 5, 7};
    vector<int> test_feature_two = {1, 15, 27};
    double test_accuracy_one = leave_one_out_validation(dataset, test_feature_one);
    cout << "Accuracy using features {3, 5, 7}: " << test_accuracy_one << "%" << endl;
    double test_accuracy_two = leave_one_out_validation(dataset, test_feature_two);
    cout << "Accuracy using features {1, 15, 27}: " << test_accuracy_two << "%" << endl;
    return 0;
}
