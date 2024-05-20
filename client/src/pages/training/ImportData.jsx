/* eslint-disable no-unused-vars */
import React, { useState, useEffect } from "react";
import TableImport from "../../components/TableImport";
import { Button, Alert, message } from "antd";
import axios from "axios";

const ImportData = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [data, setData] = useState([]);
  const [showSuccessAlert, setShowSuccessAlert] = useState(false);
  const [showWarningAlert, setShowWarningAlert] = useState(false);

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

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleImportExcel = () => {
    const formData = new FormData();
    formData.append("file", selectedFile);
    setUploading(true);
    axios
      .post("http://localhost:5000/import-excel", formData)
      .then((response) => {
        if (response.status === 200) {
          getData();
          setShowSuccessAlert(true);
        } else {
          setShowWarningAlert(true);
        }
      })
      .catch((error) => {
        console.error("Error importing Excel:", error);
        setShowWarningAlert(true);
      })
      .finally(() => {
        setUploading(false);
      });
  };

  const handleDeleteExcel = () => {
    axios
      .delete("http://localhost:5000/delete-all")
      .then((response) => {
        if (response.status === 200) {
          setData([]);
          message.success("All tweets deleted successfully");
        } else {
          alert("Failed to delete tweets!");
        }
      })
      .catch((error) => {
        console.error("Error deleting tweets:", error);
        alert("Failed to delete tweets!");
      });
  };

  return (
    <div>
      {showSuccessAlert && (
        <Alert
          message="Success"
          description="Excel imported successfully!"
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
          description="Failed to import Excel!"
          type="error"
          className="mb-3"
          showIcon
          closable
          onClose={() => setShowWarningAlert(false)}
        />
      )}
      <input
        type="file"
        onChange={handleFileChange}
        style={{ color: "white" }}
      />
      <Button
        type="primary"
        onClick={handleImportExcel}
        loading={uploading}
        style={{
          marginTop: 16,
          marginBottom: 16,
          marginRight: 16,
        }}
      >
        {uploading ? "Uploading" : "Start Upload"}
      </Button>
      <Button
        type="primary"
        danger
        className="me-3"
        onClick={handleDeleteExcel}
      >
        Delete Record
      </Button>
      <TableImport data={data} itemsPerPage={5} title={"Import Data"} />
    </div>
  );
};

export default ImportData;
