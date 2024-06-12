/* eslint-disable no-unused-vars */
import React, { useState } from "react";
import { Button, message, Alert } from "antd";
import { FileSyncOutlined } from "@ant-design/icons";
import axios from "axios";

const Modeling = () => {
  const [predictedSentimentTraining, setPredictedSentimentTraining] = useState(null);
  const [predictedSentimentTesting, setPredictedSentimentTesting] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleProcess = async () => {
    setLoading(true);
    try {
      const responseTraining = await axios.post(
        "http://localhost:5000/predict-sentiment"
      );
      const responseTesting = await axios.post(
        "http://localhost:5000/predict-sentiment-testing"
      );
      if ((responseTraining.status == 200 && responseTesting.status == 200) && (predictedSentimentTraining != 0 && predictedSentimentTesting != 0)) {
        setPredictedSentimentTraining(responseTraining.data);
        setPredictedSentimentTesting(responseTesting.data);
        message.success("Sentiment predicted successfully");
      } else {
        message.error("Data is empty!")
      }
    } catch (error) {
      console.error("Error predicting sentiment", error);
      message.error("Failed to predict sentiment");
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    try {
      setLoading(true);
      const response = await axios.post(
        "http://localhost:5000/export-tf",
        {},
        { responseType: "blob" }
      );
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "tf_results.csv");
      document.body.appendChild(link);
      link.click();
      message.success("TF exported successfully!");
    } catch (error) {
      console.error("Error export csv", error);
      message.error("Failed to export csv");
    } finally {
      setLoading(false);
    }
  };

  const handleExportIDF = async () => {
    try {
      setLoading(true);
      const response = await axios.post(
        "http://localhost:5000/export-idf",
        {},
        { responseType: "blob" }
      );
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "idf_results.csv");
      document.body.appendChild(link);
      link.click();
      message.success("IDF exported successfully!");
    } catch (error) {
      console.error("Error exporting IDF to CSV", error);
      message.error("Failed to export IDF to CSV");
    } finally {
      setLoading(false);
    }
  };

  const handleExportTFIDF = async () => {
    try {
      setLoading(true);
      const response = await axios.post(
        "http://localhost:5000/export-tfidf",
        {},
        { responseType: "blob" }
      );
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "tfidf_results.csv");
      document.body.appendChild(link);
      link.click();
      message.success("TF-IDF exported successfully!");
    } catch (error) {
      console.error("Error exporting TF-IDF to CSV", error);
      message.error("Failed to export TF-IDF to CSV");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {(predictedSentimentTraining == null ||
        predictedSentimentTesting == null) && (
        <h2 style={{ color: "white" }} className="mb-3">
          Modeling with KNN + TF-IDF
        </h2>
      )}
      <Button
        type="primary"
        className="mb-3"
        icon={<FileSyncOutlined />}
        loading={loading}
        onClick={handleProcess}
      >
        Process
      </Button>

      <Button
        type="primary"
        className="mx-3"
        icon={<FileSyncOutlined />}
        loading={loading}
        onClick={handleExport}
      >
        TF Result
      </Button>

      <Button
        type="primary"
        className=""
        icon={<FileSyncOutlined />}
        loading={loading}
        onClick={handleExportIDF}
      >
        IDF Result
      </Button>

      <Button
        type="primary"
        className="mx-3"
        icon={<FileSyncOutlined />}
        loading={loading}
        onClick={handleExportTFIDF}
      >
        TF-IDF Result
      </Button>

      {predictedSentimentTraining !== null && (
        <Alert
          message="Predicted Sentiment With KNN Model (Training Data)"
          description={`Positive Count: ${predictedSentimentTraining.positive_count} | Negative Count: ${predictedSentimentTraining.negative_count} | Neutral Count: ${predictedSentimentTraining.neutral_count}`}
          type="info"
          showIcon
        />
      )}

      <br></br>

      {predictedSentimentTesting !== null && (
        <Alert
          message="Predicted Sentiment With KNN Model (Testing Data)"
          description={`Positive Count: ${predictedSentimentTesting.positive_count} | Negative Count: ${predictedSentimentTesting.negative_count} | Neutral Count: ${predictedSentimentTesting.neutral_count}`}
          type="info"
          showIcon
        />
      )}
    </div>
  );
};

export default Modeling;
