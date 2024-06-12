/* eslint-disable no-unused-vars */
import React, { useState } from "react";
import { Button, message, Alert } from "antd";
import { FileSyncOutlined } from "@ant-design/icons";
import axios from "axios";

const Evaluate = () => {
  const [evaluationDataTraining, setEvaluationDataTraining] = useState(null);
  const [evaluationDataTesting, setEvaluationDataTesting] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleProcess = async () => {
    try {
      setLoading(true);
      const responseTraining = await axios.post(
        "http://localhost:5000/calculate-accuracy"
      );
      const responseTesting = await axios.post(
        "http://localhost:5000/calculate-accuracy-testing"
      );
      setEvaluationDataTraining(responseTraining.data);
      setEvaluationDataTesting(responseTesting.data);
      message.success("Accuracy calculated!");
    } catch (error) {
      console.error("Error calculating accuracy!", error);
      message.error("Data is empty!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {(evaluationDataTraining == null || evaluationDataTesting == null) && (
        <h2 style={{ color: "white" }} className="mb-3">
          Evaluate Model
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
      {evaluationDataTraining !== null && (
        <Alert
          message="Accuration Training Data"
          description={
            <div>
              <h2>Confusion Matrix:</h2>
              <table>
                <tbody>
                  {evaluationDataTraining.confusion_matrix.map(
                    (row, rowIndex) => (
                      <tr key={rowIndex}>
                        {row.map((value, colIndex) => (
                          <td key={colIndex}>{value}</td>
                        ))}
                      </tr>
                    )
                  )}
                </tbody>
              </table>
              <hr></hr>
              <h2>Evaluation Metrics:</h2>
              <p>Precision: {evaluationDataTraining.precision}</p>
              <p>Recall: {evaluationDataTraining.recall}</p>
              <p>F1 Score: {evaluationDataTraining.f1_score}</p>
              <hr></hr>
              <h2>Accuracy: {evaluationDataTraining.accuracy}</h2>
            </div>
          }
          type="info"
          showIcon
        />
      )}

      <br></br>
      
      {evaluationDataTesting !== null && (
        <Alert
          message="Accuration Testing Data"
          description={
            <div>
              <h2>Confusion Matrix:</h2>
              <table>
                <tbody>
                  {evaluationDataTesting.confusion_matrix.map(
                    (row, rowIndex) => (
                      <tr key={rowIndex}>
                        {row.map((value, colIndex) => (
                          <td key={colIndex}>{value}</td>
                        ))}
                      </tr>
                    )
                  )}
                </tbody>
              </table>
              <hr></hr>
              <h2>Evaluation Metrics:</h2>
              <p>Precision: {evaluationDataTesting.precision}</p>
              <p>Recall: {evaluationDataTesting.recall}</p>
              <p>F1 Score: {evaluationDataTesting.f1_score}</p>
              <hr></hr>
              <h2>Accuracy: {evaluationDataTesting.accuracy}</h2>
            </div>
          }
          type="info"
          showIcon
        />
      )}
    </div>
  );
};

export default Evaluate;
