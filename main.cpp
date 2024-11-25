#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>  // For sqrt and pow
#include <algorithm>  // For transform
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

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
        // Leave out instance `i` for testing
        vector<Instance> training_data = dataset;
        Instance test_instance = dataset[i];
        training_data.erase(training_data.begin() + i);

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
        istringstream iss(line);
        Instance instance;
        iss >> instance.class_label;
        double value;
        while (iss >> value) {
            instance.features.push_back(value);
        }
        dataset.push_back(instance);
    }

    // Normalize features
    size_t num_features = dataset[0].features.size();
    for (size_t i = 0; i < num_features; ++i) {
        double min_val = numeric_limits<double>::max();
        double max_val = numeric_limits<double>::min();

        for (const auto& instance : dataset) {
            min_val = min(min_val, instance.features[i]);
            max_val = max(max_val, instance.features[i]);
        }

        for (auto& instance : dataset) {
            instance.features[i] = (instance.features[i] - min_val) / (max_val - min_val);
        }
    }

    return dataset;
}

double real_evaluation_function(const vector<Instance>& dataset, const vector<int>& feature_subset) {
    return leave_one_out_validation(dataset, feature_subset);
}

// Forward Selection
void forward_selection(int num_features, const vector<Instance>& dataset) {
    cout << "Running Forward Selection Algorithm..." << endl;

    vector<int> best_features;
    double best_accuracy = 0.0;

    // Variables to track the best overall feature subset
    vector<int> best_overall_features;
    double best_overall_accuracy = 0.0;

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

    // Print the best overall feature set and accuracy
    cout << "Finished search!! The best feature subset is { ";
    for (int f : best_overall_features) cout << f << " ";
    cout << "} with an accuracy of " << best_overall_accuracy << "%" << endl;
}

// Backward Elimination
void backward_elimination(int num_features, const vector<Instance>& dataset) {
    cout << "Running Backward Elimination Algorithm..." << endl;

    vector<int> current_features(num_features);
    for (int i = 0; i < num_features; ++i) current_features[i] = i + 1;

    double best_accuracy = leave_one_out_validation(dataset, current_features);
    vector<int> best_features = current_features;

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

            cout << "Best feature set so far: { ";
            for (int f : best_features) cout << f << " ";
            cout << "} with accuracy " << best_accuracy << "%" << endl << endl;
        }
    }

    cout << "Finished search!! The best feature subset is { ";
    for (int f : best_features) cout << f << " ";
    cout << "} with an accuracy of " << best_accuracy << "%" << endl;
}


int main() {
    srand(static_cast<unsigned int>(time(0)));  // Seed random number generator
    
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
    vector<int> test_features = {3, 5, 7};
    double test_accuracy = leave_one_out_validation(dataset, test_features);
    cout << "Accuracy using features {3, 5, 7}: " << test_accuracy << "%" << endl;

    return 0;
}
