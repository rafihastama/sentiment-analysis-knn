/* eslint-disable no-unused-vars */
import React, { useState, useEffect } from "react";
import TableLabelingTesting from "../../components/TableLabelingTesting";
import { Button, message } from "antd";
import { DeliveredProcedureOutlined } from "@ant-design/icons";
import axios from "axios";

const LabelingTesting = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getData();
  }, []);

  const getData = async () => {
    try {
      const response = await axios.get("http://localhost:5000/get-data-testing");
      setData(response.data);
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  };

  const labelSentimentAutomatically = async () => {
    try {
      setLoading(true);
      const response = await axios.post("http://localhost:5000/label-sentiment-automatically-testing");
      if (response.status === 200) {
        getData();
        message.success("Sentiment labeled automatically!");
      } else {
        message.error("Failed to label sentiment automatically!");
      }
    } catch (error) {
      console.error("Error labeling sentiment automatically:", error);
      message.error("Failed to label sentiment automatically!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <Button
        type="primary"
        className="mb-3 ml-3"
        icon={<DeliveredProcedureOutlined />}
        loading={loading}
        onClick={labelSentimentAutomatically}
      >
        Label Sentiment Automatically
      </Button>
      <TableLabelingTesting
        data={data}
        itemsPerPage={12}
        title={"Labeling Testing"}
      />
    </div>
  );
};

export default LabelingTesting;
