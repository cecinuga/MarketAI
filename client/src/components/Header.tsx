import { Col, Container, Row } from "react-bootstrap";

export default function Header(){
    return (
        <>
            <Container >
                <Row>
                    <Col col={12}>
                        <div className="fs-4 fw-semibold">MarketAI</div>
                    </Col>
                </Row>
                <Row>
                    <Col col={12} className="text-center">

                    </Col>
                </Row>
            </Container>
        </>
    )
}