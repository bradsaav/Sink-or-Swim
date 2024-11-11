#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

// returns a random accuracy for a given set of features
double stub_evaluation_function(const vector<int>& features) {
    return (rand() % 10000) / 100.0;
}

void forward_selection(int num_features) {
    cout << "Running Forward Selection Algorithm..." << endl;

    // Display the accuracy with no features selected
    vector<int> no_features;
    double no_feature_accuracy = stub_evaluation_function(no_features);
    cout << "Using no features and “random” evaluation, I get an accuracy of " << no_feature_accuracy << "%" << endl;

    vector<int> best_features;
    double best_accuracy = no_feature_accuracy;  // Initialize with no-feature accuracy
    vector<int> overall_best_features = best_features;
    double overall_best_accuracy = best_accuracy;

    cout << "\nStarting search with no features selected." << endl;
    for (int i = 0; i < num_features; ++i) {
        double curr_best_accuracy = 0.0;
        int curr_best_feature = -1;

        for (int feature = 1; feature <= num_features; ++feature) {
            if (find(best_features.begin(), best_features.end(), feature) == best_features.end()) {
                vector<int> curr_features = best_features;
                curr_features.push_back(feature);
                double accuracy = stub_evaluation_function(curr_features);
                cout << "Using features { ";
                for (int f : curr_features) cout << f << " ";
                cout << "} accuracy is " << accuracy << "%" << endl;

                if (accuracy > curr_best_accuracy) {
                    curr_best_accuracy = accuracy;
                    curr_best_feature = feature;
                }
            }
        }

        if (curr_best_feature != -1) {
            best_features.push_back(curr_best_feature);
            best_accuracy = curr_best_accuracy;

            // Update overall best if current is the best
            if (best_accuracy > overall_best_accuracy) {
                overall_best_accuracy = best_accuracy;
                overall_best_features = best_features;
            }

            cout << "Best feature set so far: { ";
            for (int f : best_features) cout << f << " ";
            cout << "} with accuracy " << best_accuracy << "%" << endl << endl;
        }
    }

    // Final check for accuracy decrease
    if (best_accuracy < overall_best_accuracy) {
        cout << "(Warning, Accuracy has decreased!)" << endl;
    }

    cout << "Finished search!! The best feature subset is { ";
    for (int f : overall_best_features) cout << f << " ";
    cout << "} with an accuracy of " << overall_best_accuracy << "%" << endl;
}

int main() {
    srand(static_cast<unsigned int>(time(0)));  // Seed random number generator

    int num_features;
    cout << "Enter the total number of features: ";
    cin >> num_features;

    int choice;
    cout << "Choose an algorithm to run:\n1. Forward Selection\n2. Backward Elimination\n";
    cin >> choice;

    if (choice == 1) {
        forward_selection(num_features);
    } else if (choice == 2) {
        //backward_elimination(num_features);
    } else {
        cout << "Invalid choice!" << endl;
    }

    return 0;
}
