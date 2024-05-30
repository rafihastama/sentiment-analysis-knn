/* eslint-disable no-unused-vars */
import React, { useState, useEffect } from "react";
import TablePreprocessingTrain from "../components/TablePreprocessingTrain";
import { Button, Alert } from "antd";
import { FileSyncOutlined } from "@ant-design/icons";
import axios from "axios";

const Preprocessing = () => {
  const [processedData, setProcessedData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showSuccessAlert, setShowSuccessAlert] = useState(false);
  const [showWarningAlert, setShowWarningAlert] = useState(false);

  useEffect(() => {
    getProcessedData();
  }, []);

  const getProcessedData = async () => {
    try {
      const response = await axios.get("http://localhost:5000/get-processed-data");
      setProcessedData(response.data);
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  };

  const handleProcessClick = async () => {
    setLoading(true);
    try {
      const response = await axios.post(
        "http://localhost:5000/preprocess-tweets"
      );
      if (response.status === 200) {
        getProcessedData();
        setShowSuccessAlert(true);
      } else {
        setShowWarningAlert(true);
      }
    } catch (error) {
      console.error("Error processing tweets:", error);
      setShowWarningAlert(true);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {showSuccessAlert && (
        <Alert
          message="Success"
          description="Tweets processed successfully!"
          type="success"
          className="mb-3"
          showIcon
          closable
          onClose={() => setShowSuccessAlert(false)}
        />
      )}
      {showWarningAlert && (
        <Alert
          message="Error"
          description="Failed to process tweets!"
          type="error"
          className="mb-3"
          showIcon
          closable
          onClose={() => setShowWarningAlert(false)}
        />
      )}
      <Button
        type="primary"
        className="mb-3"
        icon={<FileSyncOutlined />}
        loading={loading}
        onClick={handleProcessClick}
      >
        Process
      </Button>
      <TablePreprocessingTrain
        data={processedData}
        itemsPerPage={5}
        title={"Preprocessing"}
      />
    </div>
  );
};

export default Preprocessing;
