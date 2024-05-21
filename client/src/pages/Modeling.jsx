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
    try {
      setLoading(true);
      const responseTraining = await axios.post(
        "http://localhost:5000/predict-sentiment"
      );
      const responseTesting = await axios.post(
        "http://localhost:5000/predict-sentiment-testing"
      );
      setPredictedSentimentTraining(responseTraining.data);
      setPredictedSentimentTesting(responseTesting.data);
      message.success("Sentiment predicted successfully");
    } catch (error) {
      console.error("Error predicting sentiment", error);
      message.error("Failed to predict sentiment");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {(predictedSentimentTraining == null ||
        predictedSentimentTesting == null) && (
        <h2 style={{ color: "white" }} className="mb-3">
          Modeling with KNN
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
