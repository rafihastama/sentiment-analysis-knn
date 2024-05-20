/* eslint-disable no-unused-vars */
/* eslint-disable react/prop-types */
import React, { useState } from 'react';
import { Table, Pagination, PaginationItem, PaginationLink, Card,
  CardHeader,
  CardBody,
  CardTitle,
  Row,
  Col,
} from 'reactstrap';

function TablePreprocessingTrain({ data, itemsPerPage, title }) {
    const [currentPage, setCurrentPage] = useState(0);
    const totalData = data.length;
    const totalPages = Math.ceil(totalData / itemsPerPage);
    const maxDisplayedPages = 5;

    const handleClick = (e, index) => {
        e.preventDefault();
        setCurrentPage(index);
    };

    const renderTableData = () => {
        const start = currentPage * itemsPerPage;
        const end = start + itemsPerPage;
        return data.slice(start, end).map((item, index) => (
          <tr key={index}>
            <td>{item.id}</td>
            <td>{item.username}</td>
            <td>{item.full_text}</td>
            <td>{item.cleaned_text}</td>
            <td>{item.tokenized_words}</td>
            <td>
              {item.formal_text &&
                item.stopword_removal &&
                item.formal_text.split(" ").map((word, idx) => (
                  <span
                    key={idx}
                    style={{
                      color: item.stopword_removal.includes(word)
                        ? "black"
                        : "blue",
                    }}
                  >
                    {word}{" "}
                  </span>
                ))}
            </td>
            <td>{item.stopword_removal}</td>
            <td>{item.processed_text}</td>
          </tr>
        ));
    };

    const renderPagination = () => {
        const paginationItems = [];
        const startPage = Math.max(0, currentPage - Math.floor(maxDisplayedPages / 2));
        const endPage = Math.min(totalPages - 1, startPage + maxDisplayedPages - 1);
        for (let i = startPage; i <= endPage ; i++) {
            paginationItems.push(
                <PaginationItem key={i} active={i === currentPage}>
                    <PaginationLink onClick={(e) => handleClick(e, i)} href="#">
                        {i + 1}
                    </PaginationLink>
                </PaginationItem>
            );
        }
        return paginationItems;
    };

    return (
        <div>
          <Row>
            <Col md="12">
              <Card>
                <CardHeader>
                  <CardTitle tag="h4">{title}</CardTitle>
                </CardHeader>
                <CardBody>
                  <Table className="tablesorter" responsive>
                    <thead className="text-primary">
                      <tr>
                        <th>No</th>
                        <th>Username</th>
                        <th>Full Text</th>
                        <th>Cleaned Text</th>
                        <th>Tokenized Words</th>
                        <th>Formal Text</th>
                        <th>Stopword Removal</th>
                        <th>Stemmed Text</th>
                      </tr>
                    </thead>
                    <tbody>{renderTableData()}</tbody>
                  </Table>
                </CardBody>
              </Card>
            </Col>
          </Row>
            <Pagination>
              <PaginationItem disabled={currentPage <= 0}>
                    <PaginationLink
                        onClick={(e) => handleClick(e, 0)}
                        first
                        href="#" />
                </PaginationItem>
                <PaginationItem disabled={currentPage <= 0}>
                    <PaginationLink
                        onClick={(e) => handleClick(e, currentPage - 1)}
                        previous
                        href="#" />
                </PaginationItem>

                {renderPagination()}

                <PaginationItem disabled={currentPage >= totalPages - 1}>
                    <PaginationLink
                        onClick={(e) => handleClick(e, currentPage + 1)}
                        next
                        href="#" />
                </PaginationItem>
                <PaginationItem disabled={currentPage >= totalPages - 1}>
                    <PaginationLink
                        onClick={(e) => handleClick(e, totalPages - 1)}
                        last
                        href="#" />
                </PaginationItem>
            </Pagination>
        </div>
    );
}

export default TablePreprocessingTrain;
