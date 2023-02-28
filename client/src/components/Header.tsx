import { Col, Container, Row } from "react-bootstrap";
import NavBar from "./NavBar";

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
                        <NavBar />
                    </Col>
                </Row>
            </Container>
        </>
    )
}