from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from io import BytesIO
import base64
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


# Route for displaying the dataset
@app.route('/data-show')
def data_show():
    # Load dataset from file or database
    dataset = pd.read_csv('datasets/segmentation data.csv')
    # Get feature names
    feature_names = dataset.columns.tolist()
    # Convert dataset to HTML table
    table_html = dataset.to_html(index=False)
    return render_template('data_show.html', table_html=table_html, feature_names=feature_names)


# Route for generating and displaying clustering results
@app.route('/generate-clusters', methods=['GET', 'POST'])
def generate_clusters():
    if request.method == 'POST':
        # Get form data
        num_clusters = int(request.form['num_clusters'])
        algo_type = request.form['algo_type']

        # Load dataset from file or database
        dataset = pd.read_csv('datasets/segmentation data.csv')

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
            plot_dir = 'static/plots/'
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
                    # Encode plot image to base64 for embedding in HTML
                    image_names.append(plot_file)



            # Return results HTML
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
