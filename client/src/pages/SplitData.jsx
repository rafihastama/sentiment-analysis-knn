/* eslint-disable no-unused-vars */
import React, { useState, useEffect } from "react"
import { Button, message, Card, Row, Col } from "antd"
import { SplitCellsOutlined } from "@ant-design/icons"
import axios from "axios"

function SplitData() {
  const [data, setData] = useState({ training_count: 0, testing_count: 0 })
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    getData()
  }, [])

  const getData = async () => {
    try {
      const response = await axios.get("http://localhost:5000/get-both-data")
      setData(response.data)
    } catch (error) {
      console.error("Error Fetching Data", error)
    }
  }

  const handleProcessClick = async () => {
    setLoading(true)
    try {
      const response = await axios.post("http://localhost:5000/split-data")
      if (response.status === 200 && data != 0) {
        getData()
        message.success("Split Data Success")
      } else {
        message.error("Data is empty!")
      }
    } catch (error) {
      console.error("Error Splitting Data", error)
      message.error("Data is empty!")
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteData = () => {
    axios
      .delete("http://localhost:5000/delete-both")
      .then((response) => {
        if (response.status === 200) {
          setData({ training_count: 0, testing_count: 0 })
          message.success("All tweets deleted successfully!")
        } else if (response.status === 204) {
          message.error("Data is empty!")
        }
      })
      .catch((error) => {
        console.error("Error deleting tweets:", error)
        message.error("Failed to delete tweets!")
      })
  }

  return (
    <div>
      <h2 style={{ color: "white" }} className="mb-3">
        Split Data
      </h2>
      <Button
        type="primary"
        className="mb-3"
        icon={<SplitCellsOutlined />}
        loading={loading}
        onClick={handleProcessClick}
      >
        Split
      </Button>
      <Button
        type="primary"
        danger
        className="me-3 mx-3"
        onClick={handleDeleteData}
      >
        Delete Record
      </Button>
      <Row gutter={16}>
        <Col span={12}>
          <center>
            <Card title="Training Data">
              <center>
                <h1>{data.training_count}</h1>
              </center>
            </Card>
          </center>
        </Col>
        <Col span={12}>
          <center>
            <Card title="Testing Data">
              <center>
                <h1>{data.testing_count}</h1>
              </center>
            </Card>
          </center>
        </Col>
      </Row>
    </div>
  )
}

export default SplitData
