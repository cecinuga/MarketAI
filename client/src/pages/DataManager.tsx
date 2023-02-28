import axios from "axios";
import { Button } from "react-bootstrap";
import { useQuery } from "react-query";
import Header from "../components/Header";
import { Api } from "../lib/api";

export default function DataManager(){

    const {data, isLoading} = useQuery("datamanager", async () => {
        const res = await Api.post("datamanager");
        return res;
    })

    return (
        <>
            <Header />
        </>
    )
}