from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os


app = Flask(__name__)


# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')


# Route for collecting form data via wifi tunneling
@app.route('/wifi-tunneling', methods=['GET', 'POST'])
def wifi_tunneling():
    return render_template('wifi_tunneling.html')


# Route for generating and displaying clustering results
@app.route('/generate-clusters', methods=['GET', 'POST'])
def generate_clusters():
    if request.method == 'POST':
        # Get form data
        num_clusters = int(request.form['num_clusters'])
        algo_type = request.form['algo_type']

        print("--GOT ALGO--")
        dataset_selection = request.form['dataset']
        print("--GOT DATASET--")

        # Determine dataset selection and load dataset
        if dataset_selection == 'dataset1':
            dataset = pd.read_csv('datasets/Sales Transaction.csv')
        elif dataset_selection == 'dataset2':
            dataset = pd.read_csv('datasets/Personal Info.csv')
        elif dataset_selection == 'dataset3':
            dataset = pd.read_csv('datasets/Customer Demographics.csv')
        elif dataset_selection == 'dataset4':
            dataset = pd.read_csv('datasets/Wine Quality.csv')
        elif dataset_selection == 'dataset5':
            dataset = pd.read_csv('datasets/online retail.csv')
        elif dataset_selection == 'dataset6':
            dataset = pd.read_csv('datasets/geolocation.csv')
        elif dataset_selection == 'dataset7':
            dataset = pd.read_csv('datasets/Product Sales.csv')
        elif dataset_selection == 'dataset8':
            dataset = pd.read_csv('datasets/Sports Performance.csv')
        elif dataset_selection == 'dataset9':
            dataset = pd.read_csv('datasets/Customer-Segmentation.csv')
        elif dataset_selection == 'dataset10':
            dataset = pd.read_csv('dataset7.csv')
        else :
            return render_template('error.html')

        # Identify and remove non-numeric columns
        non_numeric_columns = []
        for column in dataset.columns:
            if not np.issubdtype(dataset[column].dtype, np.number):
                non_numeric_columns.append(column)

        # Drop rows with NaN values
        dataset = dataset.dropna()
        dataset = dataset.drop(columns=non_numeric_columns)

        # Convert remaining columns to floats
        dataset = dataset.astype(float)


        # Get feature names
        feature_names_a = dataset.columns.tolist()
        # Convert dataset to HTML table
        table_html = dataset.to_html(index=False)


        # Run clustering algorithm
        if algo_type == 'k-means':
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(dataset)
            dataset['cluster'] = kmeans.labels_
            # Get feature names
            feature_names = dataset.columns.tolist()
            # Remove 'cluster' column from feature names
            feature_names.remove('cluster')
            # Generate scatter plot
            plots = []
            # Generate scatter plot
            plots = []
            plot_dir = os.path.join('static', 'plots')
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            image_names = []
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    fig, ax = plt.subplots()
                    ax.scatter(dataset[feature_names[i]], dataset[feature_names[j]], c=dataset['cluster'])
                    ax.set_xlabel(feature_names[i])
                    ax.set_ylabel(feature_names[j])
                    # Save plot image to file
                    plot_file = f'plot_{feature_names[i]}_{feature_names[j]}.png'
                    plot_path = os.path.join(plot_dir, plot_file)
                    plt.savefig(plot_path, format='png')
                    plt.close(fig)
                    image_names.append(plot_file)

            # Return results HTM
            return render_template('result.html', image_names=image_names, feature_names=feature_names,feature_names_a=feature_names_a  ,table_html=table_html)

        elif algo_type == 'hierarchical':
            # Implement hierarchical clustering algorithm
            # ...
            return render_template('error.html')
        elif algo_type == 'DBSCAN':
            # Implement DBSCAN clustering algorithm
            # ...
            return render_template('error.html')
        else:
            return render_template('error.html')
    return render_template('error.html')



if __name__ == '__main__':
    app.run()
