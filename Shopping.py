import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    with open(filename, "r") as shopping_data_file:
        file_reader = csv.reader(shopping_data_file)
        evidence_index = 0

        evidence = []
        labels = []

        # Go into every item in file Reader and get the label and the evidence
        first = 0
        for item in file_reader:
            if first == 0:
                first = 1
                continue

            # Seperate the laben & evidence
            label = item.pop()

            # Convert labels to readable numbers
            if label == "FALSE":
                label = 0
            else:
                label = 1
            labels.append(label)

            # Convert ints to readable numbers
            revised_evidence = []
            index = 0
            while index < 17:

                # Convert to ints
                if index in [0, 2, 4, 11, 12, 13, 14]:
                    item[index] = int(item[index])

                # Convert to floats
                elif index in [1, 3, 5, 6, 7, 8, 9]:
                    item[index] = float(item[index])

                # Convert months to int
                elif index == 10:
                    months = ["jan", "feb", "mar", "apr", "may", "june",
                              "jul", "aug", "sep", "oct", "nov", "dec"]
                    month_index = 0
                    while month_index < 12:
                        if item[index].lower() == months[month_index]:
                            item[index] = month_index
                            break
                        month_index += 1

                # Convert visitor type to int
                elif index == 15:
                    if item[index] == "Returning_Visitor":
                        item[index] = 1
                    else:
                        item[index] = 0

                # Convert weekend/weekday to int
                else:
                    if item[index] == "FALSE":
                        item[index] = 0
                    else:
                        item[index] = 1

                revised_evidence.append(item[index])
                index += 1

            evidence.append(revised_evidence)

    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # impostors = []
    # for lists in evidence:
    #    for item in lists:
    #        if type(item) != int and type(item) != float:
    #            impostors.append(item)

    # if len(impostors) != 0:
    #    print(impostors)
    #    raise ValueError

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    specificity = 0
    sensitivity = 0
    for item in labels:
        if item == 0:
            specificity += 1
        else:
            sensitivity += 1

    correct_specificity = 0
    correct_sensitivity = 0
    index = 0
    while index < len(labels):
        if labels[index] == predictions[index]:
            if labels[index] == 1:
                correct_sensitivity += 1
            else:
                correct_specificity += 1
        index += 1

    specificity = (correct_specificity / specificity)
    sensitivity = (correct_sensitivity / sensitivity)

    return sensitivity, specificity


if __name__ == "__main__":
    main()
