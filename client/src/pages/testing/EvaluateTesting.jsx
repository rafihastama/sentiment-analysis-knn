/* eslint-disable no-unused-vars */
import React, { useState } from "react";
import { Button, message, Alert } from "antd";
import { FileSyncOutlined } from "@ant-design/icons";
import axios from "axios";

const EvaluateTesting = () => {
  const [evaluationData, setEvaluationData] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleProcess = async () => {
    try {
      setLoading(true);
      const response = await axios.post(
        "http://localhost:5000/calculate-accuracy-testing"
      );
      setEvaluationData(response.data);
      message.success("Accuracy calculated!");
    } catch (error) {
      console.error("Error calculating accuracy!", error);
      message.error("Error calculating accuracy!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {evaluationData == null && (
        <h2 style={{ color: "white" }} className="mb-3">
          Testing Evaluate Training Data
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
      {evaluationData !== null && (
        <Alert
          message="Accuration Training Data"
          description={
            <div>
              <h2>Confusion Matrix:</h2>
              <table>
                <tbody>
                  {evaluationData.confusion_matrix.map((row, rowIndex) => (
                    <tr key={rowIndex}>
                      {row.map((value, colIndex) => (
                        <td key={colIndex}>{value}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              <hr></hr>
              <h2>Evaluation Metrics:</h2>
              <p>Precision: {evaluationData.precision}</p>
              <p>Recall: {evaluationData.recall}</p>
              <p>F1 Score: {evaluationData.f1_score}</p>
              <hr></hr>
              <h2>Accuracy: {evaluationData.accuracy}</h2>
            </div>
          }
          type="info"
          showIcon
        />
      )}
    </div>
  );
};

export default EvaluateTesting;
