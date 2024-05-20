/* eslint-disable no-unused-vars */
import React, { useState } from "react";
import { Button, message, Alert } from "antd";
import { FileSyncOutlined } from "@ant-design/icons";
import axios from "axios";

const ModelingTraining = () => {
  const [predictedSentiment, setPredictedSentiment] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleProcess = async () => {
    try {
      setLoading(true);
      const response = await axios.post(
        "http://localhost:5000/predict-sentiment"
      );
      setPredictedSentiment(response.data);
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
      {predictedSentiment == null && (
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
      {predictedSentiment !== null && (
        <Alert
          message="Predicted Sentiment With KNN Model"
          description={`Positive Count: ${predictedSentiment.positive_count} | Negative Count: ${predictedSentiment.negative_count} | Neutral Count: ${predictedSentiment.neutral_count}`}
          type="info"
          showIcon
        />
      )}
    </div>
  );
};

export default ModelingTraining;
