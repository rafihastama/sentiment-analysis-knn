/* eslint-disable no-unused-vars */
import React, { useState, useEffect } from "react"
import TableImport from "../components/TableImport"
import { UploadOutlined } from '@ant-design/icons'
import { Button, Alert, message, Upload } from "antd"
import axios from "axios"

const ImportData = () => {
  const [fileList, setFileList] = useState([])
  const [uploading, setUploading] = useState(false)
  const [data, setData] = useState([])
  const [showSuccessAlert, setShowSuccessAlert] = useState(false)
  const [showWarningAlert, setShowWarningAlert] = useState(false)

  useEffect(() => {
    getData()
  }, [])

  const getData = async () => {
    try {
      const response = await axios.get("http://localhost:5000/get-data")
      setData(response.data)
    } catch (error) {
      console.error("Error fetching data:", error)
    }
  }

  const handleUpload = () => {
    const formData = new FormData()
    fileList.forEach((file) => {
      formData.append('file', file)
    })
    setUploading(true)
    axios
      .post("http://localhost:5000/import-excel", formData)
      .then((response) => {
        if (response.status === 200) {
          getData()
          setShowSuccessAlert(true)
        } else {
          setShowWarningAlert(true)
        }
      })
      .catch((error) => {
        console.error("Error importing Excel:", error)
        message.error("No data selected!")
      })
      .finally(() => {
        setUploading(false)
      })
  }

  const handleDeleteExcel = () => {
    axios
      .delete("http://localhost:5000/delete-all")
      .then((response) => {
        if ((response.status === 200) && (data != 0)) {
          setData([])
          message.success("All tweets deleted successfully!")
        } else {
          message.error("Data is empty!")
        }
      })
      .catch((error) => {
        console.error("Error deleting tweets:", error)
        alert("Failed to delete tweets!")
      })
  }

  const uploadProps = {
    onRemove: (file) => {
      setFileList((prevFileList) => {
        const index = prevFileList.indexOf(file)
        const newFileList = prevFileList.slice()
        newFileList.splice(index, 1)
        return newFileList
      })
    },
    beforeUpload: (file) => {
      setFileList((prevFileList) => [...prevFileList, file])
      return false
    },
    fileList,
  }

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
      <Upload {...uploadProps} className="custom-upload-list">
        <Button icon={<UploadOutlined />}>Select File</Button>
      </Upload>
      <Button
        type="primary"
        onClick={handleUpload}
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
  )
}

export default ImportData
