/* eslint-disable no-unused-vars */
import React, { useState } from "react";
import { Button, message } from "antd";
import { FileSyncOutlined } from "@ant-design/icons";
import axios from "axios";

function TFIDFTraining() {
  const [loading, setLoading] = useState(false);
  const [tfidfData, setTfidfData] = useState([]);

  const processTFIDF = async () => {
    setLoading(true)
    try {
      const response = await axios.post("http://localhost:5000/tf-idf");
      setTfidfData(response.data);
      message.success("TF-IDF calculated successfully!")
    } catch (error) {
      console.error("Error fetching TF-IDF data:", error);
      message.error("Failed to calculate TF-IDF")
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <Button
        type="primary"
        className="mb-3"
        icon={<FileSyncOutlined />}
        loading={loading}
        onClick={processTFIDF}
      >
        Process TF-IDF
      </Button>
      <h2 style={{ color: "white" }} className="mb-3">
        TF-IDF Training Data
      </h2>
      {tfidfData.map((item, index) => (
        <div key={index}>
          <h3>{item.word}</h3>
          <ul>
            {item.tfidf_values.map((value, idx) => (
              <li key={idx}>
                Document {idx + 1}: {value}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

export default TFIDFTraining;
