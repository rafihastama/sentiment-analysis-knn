/* eslint-disable no-unused-vars */
import React, { useState, useEffect } from "react"
import TableLabeling from "../components/TableLabeling"
import { Button, message } from "antd"
import { ExportOutlined } from "@ant-design/icons"
import { DeliveredProcedureOutlined } from "@ant-design/icons"
import axios from "axios"

const Labeling = () => {
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    getData()
  }, [])

  const getData = async () => {
    try {
      const response = await axios.get("http://localhost:5000/get-processed-data")
      setData(response.data)
    } catch (error) {
      console.error("Error fetching data:", error)
    }
  }

  const labelSentimentAutomatically = async () => {
    try {
      setLoading(true)
      const response = await axios.post(
        "http://localhost:5000/label-sentiment-automatically"
      )
      if ((response.status === 200) && (data != 0)) {
        await getData()
        message.success("Sentiment labeled automatically!")
      } else {
        message.error("Data is empty!")
      }
    } catch (error) {
      console.error("Error labeling sentiment automatically:", error)
      message.error("Failed to label sentiment automatically!")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <Button
        type="primary"
        className="mb-3 ml-3"
        icon={<DeliveredProcedureOutlined />}
        onClick={labelSentimentAutomatically}
        loading={loading}
      >
        Label Sentiment Automatically
      </Button>
      <TableLabeling data={data} itemsPerPage={12} title={"Labeling"} />
    </div>
  )
}

export default Labeling
