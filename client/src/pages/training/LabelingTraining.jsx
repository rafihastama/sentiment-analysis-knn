/* eslint-disable no-unused-vars */
import React, { useState, useEffect } from "react";
import TableLabelingTraining from "../../components/TableLabelingTraining";
import { Button, message } from "antd";
import { ExportOutlined } from "@ant-design/icons";
import { DeliveredProcedureOutlined } from "@ant-design/icons";
import axios from "axios";

const LabelingTraining = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    getData();
  }, []);

  const getData = async () => {
    try {
      const response = await axios.get("http://localhost:5000/get-data");
      setData(response.data);
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  };

  const exportData = async () => {
    try {
      const response = await axios.post(
        "http://localhost:5000/export-data",
        {},
        { responseType: "blob" }
      );
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "exported_data.csv");
      document.body.appendChild(link);
      link.click();
      message.success("Data exported successfully!");
    } catch (error) {
      message.error("Failed to export data!");
    }
  };

  const labelSentimentAutomatically = async () => {
    try {
      setLoading(true)
      const response = await axios.post(
        "http://localhost:5000/label-sentiment-automatically"
      );
      console.log(response.data.message);
      message.success("Sentiment labeled automatically!");
      getData();
    } catch (error) {
      console.error("Error labeling sentiment automatically:", error);
      message.error("Failed to label sentiment automatically!");
    } finally {
      setLoading(false)
    }
  };

  return (
    <div>
      <Button
        type="primary"
        className="mb-3"
        icon={<ExportOutlined />}
        onClick={exportData}
      >
        Export Data
      </Button>
      <Button
        type="primary"
        className="mb-3 ml-3 mx-3"
        icon={<DeliveredProcedureOutlined />}
        onClick={labelSentimentAutomatically}
        loading={loading}
      >
        Label Sentiment Automatically
      </Button>
      <TableLabelingTraining data={data} itemsPerPage={12} title={"Labeling Training"} />
    </div>
  );
};

export default LabelingTraining;
