/* eslint-disable no-unused-vars */
import React, { useState } from "react"
import { Button, message, Alert } from "antd"
import { FileSyncOutlined } from "@ant-design/icons"
import axios from "axios"

const Evaluate = () => {
  const [evaluationDataTraining, setEvaluationDataTraining] = useState(null)
  const [evaluationDataTesting, setEvaluationDataTesting] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleProcess = async () => {
    try {
      setLoading(true)
      const responseTraining = await axios.post(
        "http://localhost:5000/calculate-accuracy"
      )
      const responseTesting = await axios.post(
        "http://localhost:5000/calculate-accuracy-testing"
      )
      setEvaluationDataTraining(responseTraining.data)
      setEvaluationDataTesting(responseTesting.data)
      message.success("Accuracy calculated!")
    } catch (error) {
      console.error("Error calculating accuracy!", error)
      message.error("Data is empty!")
    } finally {
      setLoading(false)
    }
  }

  const formatPercentage = (value) => {
    return `${Math.round(value)}%`
  }

  const renderConfusionMatrix = (matrix) => {
    const labels = ["Negatif", "Netral", "Positif"]
    return (
      <table style={{ width: "100%" }}>
        <thead>
          <tr>
            <th
              style={{ border: "1px solid black", padding: "8px" }}
              colSpan="2"
              rowSpan="2"
            ></th>
            <th
              style={{ border: "1px solid black", padding: "8px" }}
              colSpan={3}
            >
              <center>Predicted</center>
            </th>
          </tr>
          <tr>
            {labels.map((label, index) => (
              <th
                key={index}
                style={{ border: "1px solid black", padding: "8px" }}
              >
                <center>{label}</center>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {rowIndex === 0 && (
                <th
                  rowSpan={3}
                  style={{
                    border: "1px solid black",
                    padding: "8px",
                    textAlign: "center",
                    verticalAlign: "middle",
                  }}
                >
                  Actual
                </th>
              )}
              <th style={{ border: "1px solid black", padding: "8px" }}>
                <center>{labels[rowIndex]}</center>
              </th>
              {row.map((value, colIndex) => (
                <td
                  key={colIndex}
                  style={{ border: "1px solid black", padding: "8px" }}
                >
                  <center>{value}</center>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    )
  }

  return (
    <div>
      <h2 style={{ color: "white" }} className="mb-3">
        Evaluate Model
      </h2>
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
              {renderConfusionMatrix(evaluationDataTraining.confusion_matrix)}
              <hr />
              <h2>Evaluation Metrics:</h2>
              <p>
                Precision: {formatPercentage(evaluationDataTraining.precision)}
              </p>
              <p>Recall: {formatPercentage(evaluationDataTraining.recall)}</p>
              <p>
                F1 Score: {formatPercentage(evaluationDataTraining.f1_score)}
              </p>
              <hr />
              <h2>
                Accuracy: {formatPercentage(evaluationDataTraining.accuracy)}
              </h2>
            </div>
          }
          type="info"
          showIcon
        />
      )}

      <br />

      {evaluationDataTesting !== null && (
        <Alert
          message="Accuration Testing Data"
          description={
            <div>
              <h2>Confusion Matrix:</h2>
              {renderConfusionMatrix(evaluationDataTesting.confusion_matrix)}
              <hr />
              <h2>Evaluation Metrics:</h2>
              <p>
                Precision: {formatPercentage(evaluationDataTesting.precision)}
              </p>
              <p>Recall: {formatPercentage(evaluationDataTesting.recall)}</p>
              <p>
                F1 Score: {formatPercentage(evaluationDataTesting.f1_score)}
              </p>
              <hr />
              <h2>
                Accuracy: {formatPercentage(evaluationDataTesting.accuracy)}
              </h2>
            </div>
          }
          type="info"
          showIcon
        />
      )}
    </div>
  )
}

export default Evaluate
