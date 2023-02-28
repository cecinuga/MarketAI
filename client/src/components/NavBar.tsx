import { Nav } from "react-bootstrap";
import { useNavigate } from "react-router";

export default function NavBar(){
    const navigate = useNavigate();

    return (
        <Nav
            activeKey="/home"
            onSelect={(selectedKey) => navigate("")}
            className="d-block"
            >
            <Nav.Item>
                <Nav.Link className="d-inline-block" href="/">HomePage</Nav.Link>
                <Nav.Link className="d-inline-block" href="/datamanager">DataManager</Nav.Link>
            </Nav.Item>
        </Nav>
    )
}