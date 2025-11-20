#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/netanim-module.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/arp-l3-protocol.h"

#include <fstream>
#include <vector>
#include <sstream>
#include <map> 
#include <set> 
#include <iostream> 
#include <iomanip>
#include <algorithm>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("DDoSSimulator");

const std::string PROJECT_ROOT = "/home/traphan/ns-3-dev/ddos-project-new";

class DDoSSimulator
{
public:
    DDoSSimulator(uint32_t nIotNodes, uint32_t nAttackers, double simTime);
    void Run();

private:
    uint32_t m_nIotNodes;
    uint32_t m_nAttackers;
    double m_simTime;

    NodeContainer m_iotNodes;
    NodeContainer m_baseStations;
    NodeContainer m_serverNode;
    NodeContainer m_allNodes;

    Ipv4InterfaceContainer m_staInterfaces;
    Ipv4Address m_serverIp;
    NetDeviceContainer m_apDevices; 

    std::vector<uint32_t> m_attackerIndices;
    std::vector<Ipv4Address> m_attackerIps;

    FlowMonitorHelper m_flowMonHelper;
    Ptr<FlowMonitor> m_monitor;

    std::ofstream m_statsFile; 
    std::map<FlowId, uint32_t> m_lastTxPackets_realtime; 
    std::ofstream m_liveStatsFile; 
    std::set<Ipv4Address> m_blockedIps; 
    std::map<FlowId, uint32_t> m_lastTxPackets_live;
    
    uint32_t m_droppedPacketsInterval;
    AnimationInterface* m_anim;

    void CreateNodes();
    void SetupMobility();
    void SetupNetwork();
    void SetupApplications();
    void SetupDDoSAttack();
    
    void SetupFlowMonitor();
    void ScheduleRealtimeStats();
    void SetupMitigation();

    void MonitorLiveFlows();          
    void CheckForBlacklistUpdates();  
    void UpdateRealtimeStats();       
    void SaveFinalResults();
    void UpdateVisualStatus(bool underAttack, bool mitigationActive);

    bool PacketDropCallback(Ptr<NetDevice> device, Ptr<const Packet> packet, uint16_t protocol, const Address& from);
};

DDoSSimulator::DDoSSimulator(uint32_t nIotNodes, uint32_t nAttackers, double simTime)
    : m_nIotNodes(nIotNodes), m_nAttackers(nAttackers), m_simTime(simTime), m_anim(nullptr), m_droppedPacketsInterval(0) {}

void DDoSSimulator::CreateNodes()
{
    m_iotNodes.Create(m_nIotNodes);
    m_baseStations.Create(2);
    m_serverNode.Create(1);

    m_allNodes.Add(m_iotNodes);
    m_allNodes.Add(m_baseStations);
    m_allNodes.Add(m_serverNode);

    Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
    while (m_attackerIndices.size() < m_nAttackers) {
        uint32_t idx = uv->GetInteger(0, m_nIotNodes - 1);
        if (std::find(m_attackerIndices.begin(), m_attackerIndices.end(), idx) == m_attackerIndices.end()) {
            m_attackerIndices.push_back(idx);
        }
    }
    std::cout << "✅ TOPOLOGY: " << m_nIotNodes << " IoT, " << m_nAttackers << " Attackers, 2 BS." << std::endl;
}

void DDoSSimulator::SetupMobility()
{
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> iotPos = CreateObject<ListPositionAllocator>();
    uint32_t half = m_nIotNodes / 2;
    double spacing = 15.0;

    for (uint32_t i = 0; i < half; ++i) {
        double x = (i % 5) * spacing;
        double y = 120.0 + (i / 5) * spacing;
        iotPos->Add(Vector(x, y, 0.0));
    }
    for (uint32_t i = half; i < m_nIotNodes; ++i) {
        uint32_t j = i - half;
        double x = (j % 5) * spacing;
        double y = (j / 5) * spacing;
        iotPos->Add(Vector(x, y, 0.0));
    }

    mobility.SetPositionAllocator(iotPos);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(m_iotNodes);

    Ptr<ListPositionAllocator> infraPos = CreateObject<ListPositionAllocator>();
    infraPos->Add(Vector(100.0, 140.0, 0.0));  
    infraPos->Add(Vector(100.0, 20.0, 0.0));   
    infraPos->Add(Vector(250.0, 80.0, 0.0));   
    
    MobilityHelper staticMobility;
    staticMobility.SetPositionAllocator(infraPos);
    staticMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    staticMobility.Install(m_baseStations);
    staticMobility.Install(m_serverNode);
}

void DDoSSimulator::SetupNetwork()
{
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    WifiMacHelper wifiMac;
    YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper wifiPhy;
    wifiPhy.SetChannel(wifiChannel.Create());

    wifiMac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(Ssid("ddos-network")));
    m_apDevices = wifi.Install(wifiPhy, wifiMac, m_baseStations);

    wifiMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(Ssid("ddos-network")));
    NetDeviceContainer staDevices = wifi.Install(wifiPhy, wifiMac, m_iotNodes);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("5Mbps")); 
    p2p.SetChannelAttribute("Delay", StringValue("2ms"));
    NetDeviceContainer p2p1 = p2p.Install(m_baseStations.Get(0), m_serverNode.Get(0));
    NetDeviceContainer p2p2 = p2p.Install(m_baseStations.Get(1), m_serverNode.Get(0));

    InternetStackHelper stack;
    stack.Install(m_allNodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    m_staInterfaces = address.Assign(staDevices); 
    address.Assign(m_apDevices); 

    address.SetBase("10.1.2.0", "255.255.255.0");
    Ipv4InterfaceContainer i2 = address.Assign(p2p1);
    m_serverIp = i2.GetAddress(1); 

    address.SetBase("10.1.3.0", "255.255.255.0");
    address.Assign(p2p2);

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
}

void DDoSSimulator::SetupApplications()
{
    uint16_t serverPort = 8080;
    PacketSinkHelper sink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), serverPort));
    ApplicationContainer sinkApp = sink.Install(m_serverNode);
    sinkApp.Start(Seconds(0.0));
    sinkApp.Stop(Seconds(m_simTime));

    for (uint32_t i = 0; i < m_nIotNodes; ++i) {
        if (std::find(m_attackerIndices.begin(), m_attackerIndices.end(), i) != m_attackerIndices.end()) continue;
        OnOffHelper onOff("ns3::UdpSocketFactory", InetSocketAddress(m_serverIp, serverPort));
        onOff.SetAttribute("DataRate", StringValue("50kbps"));
        onOff.SetAttribute("PacketSize", UintegerValue(512)); 
        onOff.SetAttribute("OnTime", StringValue("ns3::ExponentialRandomVariable[Mean=2.0]"));
        onOff.SetAttribute("OffTime", StringValue("ns3::ExponentialRandomVariable[Mean=1.0]"));
        ApplicationContainer app = onOff.Install(m_iotNodes.Get(i));
        app.Start(Seconds(1.0 + i * 0.2));
        app.Stop(Seconds(m_simTime - 1.0));
    }
}

void DDoSSimulator::SetupDDoSAttack()
{
    uint16_t attackPort = 8080;
    for (uint32_t idx : m_attackerIndices) {
        OnOffHelper attack("ns3::UdpSocketFactory", InetSocketAddress(m_serverIp, attackPort));
        attack.SetAttribute("DataRate", StringValue("5000kbps"));
        attack.SetAttribute("PacketSize", UintegerValue(1024));
        attack.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=100]"));
        attack.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        ApplicationContainer app = attack.Install(m_iotNodes.Get(idx));
        app.Start(Seconds(5.0));
        app.Stop(Seconds(m_simTime - 5.0));
        m_attackerIps.push_back(m_staInterfaces.GetAddress(idx));
    }
}

void DDoSSimulator::SetupFlowMonitor() { m_monitor = m_flowMonHelper.InstallAll(); }

void DDoSSimulator::ScheduleRealtimeStats()
{
    std::string path = PROJECT_ROOT + "/data/raw/realtime_stats.csv";
    m_statsFile.open(path.c_str(), std::ofstream::out | std::ofstream::trunc); 
    m_statsFile << "time,normal_packets,attack_packets,total_throughput,avg_delay\n";
    m_statsFile.close();
    for (double t = 1.0; t <= m_simTime; t += 1.0) 
        Simulator::Schedule(Seconds(t), &DDoSSimulator::UpdateRealtimeStats, this);
}

void DDoSSimulator::UpdateVisualStatus(bool underAttack, bool mitigationActive)
{
    if (!m_anim) return;

    // --- Base Stations ---
    uint8_t r, g, b;
    std::string desc;
    double size;

    if (mitigationActive) { r=0; g=255; b=0; desc="MITIGATED"; size=5.0; }
    else if (underAttack) { r=255; g=0; b=0; desc="!!! ATTACK !!!"; size=7.0; }
    else { r=0; g=191; b=255; desc="Gateway"; size=4.0; }

    for (uint32_t i = 0; i < m_baseStations.GetN(); ++i) {
        m_anim->UpdateNodeColor(m_baseStations.Get(i), r, g, b);
        m_anim->UpdateNodeDescription(m_baseStations.Get(i), desc);
        m_anim->UpdateNodeSize(m_baseStations.Get(i), size, size);
    }

    // --- IoT nodes heatmap ---
    if (!m_monitor) return;
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(m_flowMonHelper.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = m_monitor->GetFlowStats();

    for (uint32_t i = 0; i < m_iotNodes.GetN(); ++i) {
        Ptr<Node> n = m_iotNodes.Get(i);
        uint32_t nodePackets = 0;
        for (auto it = stats.begin(); it != stats.end(); ++it) {
            Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(it->first);
            if (t.sourceAddress == m_staInterfaces.GetAddress(i)) { nodePackets = it->second.txPackets; break; }
        }
        uint8_t heat = std::min(255u, nodePackets / 10); // scale for visualization
        m_anim->UpdateNodeColor(n, heat, 255 - heat, 0);
    }

    // --- Server node color proportional throughput ---
    double totalTput = 0;
    for (auto it = stats.begin(); it != stats.end(); ++it) totalTput += it->second.rxBytes * 8.0 / 1000.0;
    uint8_t s_heat = std::min(255u, (uint32_t)(totalTput/1000));
    m_anim->UpdateNodeColor(m_serverNode.Get(0), 128, s_heat, 255 - s_heat);
}

void DDoSSimulator::UpdateRealtimeStats()
{
    if (!m_monitor) return;
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(m_flowMonHelper.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = m_monitor->GetFlowStats();

    uint32_t norm=0, attk=0;
    double tput=0;

    for (auto it = stats.begin(); it != stats.end(); ++it) {
        if (it->second.txPackets == 0) continue;
        tput += it->second.rxBytes * 8.0 / 1000.0;
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(it->first);
        bool isAtk = std::find(m_attackerIps.begin(), m_attackerIps.end(), t.sourceAddress) != m_attackerIps.end();

        uint32_t now = it->second.txPackets;
        uint32_t last = m_lastTxPackets_realtime[it->first];
        uint32_t delta = (now > last) ? now - last : 0;

        if (isAtk) attk += delta; else norm += delta;
        m_lastTxPackets_realtime[it->first] = now;
    }

    double now = Simulator::Now().GetSeconds();
    UpdateVisualStatus(attk>100, m_droppedPacketsInterval>0);
    m_droppedPacketsInterval=0;

    if (attk>0 || norm>0)
        std::cout << "STATS: T=" << std::setw(4) << now << " | Norm=" << norm << " | Attk=" << attk << std::endl;

    std::string path = PROJECT_ROOT + "/data/raw/realtime_stats.csv";
    m_statsFile.open(path.c_str(), std::ofstream::app);
    m_statsFile << now << "," << norm << "," << attk << "," << tput << ",0\n";
    m_statsFile.close();
}

bool DDoSSimulator::PacketDropCallback(Ptr<NetDevice> device, Ptr<const Packet> packet, uint16_t protocol, const Address& from)
{
    if (protocol==0x0800) {
        Ptr<Packet> p_copy = packet->Copy();
        Ipv4Header header;
        if (p_copy->PeekHeader(header)) {
            if (m_blockedIps.count(header.GetSource())) { m_droppedPacketsInterval++; return true; }
        }
        Ptr<Ipv4L3Protocol> ipv4l3 = device->GetNode()->GetObject<Ipv4L3Protocol>();
        if (ipv4l3) { ipv4l3->Receive(device, packet, protocol, from, device->GetAddress(), NetDevice::PACKET_HOST); return true; }
    } else if (protocol==0x0806) {
        Ptr<ArpL3Protocol> arp = device->GetNode()->GetObject<ArpL3Protocol>();
        if (arp) { arp->Receive(device, packet, protocol, from, device->GetAddress(), NetDevice::PACKET_HOST); return true; }
    }
    return false;
}

void DDoSSimulator::SetupMitigation()
{
    for (uint32_t i=0;i<m_apDevices.GetN();++i) {
        m_apDevices.Get(i)->SetReceiveCallback(MakeCallback(&DDoSSimulator::PacketDropCallback,this));
    }
    std::string path = PROJECT_ROOT + "/data/live/live_flow_stats.csv";
    m_liveStatsFile.open(path.c_str(), std::ofstream::out | std::ofstream::trunc);
    m_liveStatsFile << "time,source_ip,protocol,tx_packets,rx_packets,tx_bytes,rx_bytes,delay_sum,jitter_sum,lost_packets,packet_loss_ratio,throughput,flow_duration,label\n";
    m_liveStatsFile.close();
    Simulator::Schedule(Seconds(1.0), &DDoSSimulator::MonitorLiveFlows, this);
    Simulator::Schedule(Seconds(1.5), &DDoSSimulator::CheckForBlacklistUpdates, this);
}

void DDoSSimulator::MonitorLiveFlows() { /* same as previous, omitted for brevity */ }
void DDoSSimulator::CheckForBlacklistUpdates() { /* same as previous, omitted for brevity */ }
void DDoSSimulator::SaveFinalResults() { /* logic to save final results */ }

void DDoSSimulator::Run()
{
    CreateNodes();
    SetupMobility();
    SetupNetwork();
    SetupApplications();
    SetupDDoSAttack();

    AnimationInterface anim("ddos-animation.xml");
    m_anim = &anim;
    anim.EnablePacketMetadata(false);
    anim.SetMaxPktsPerTraceFile(300000);

    // Initial visualization
    for (uint32_t i=0;i<m_iotNodes.GetN();++i) {
        Ptr<Node> n=m_iotNodes.Get(i);
        bool isAtk = std::find(m_attackerIndices.begin(), m_attackerIndices.end(), i) != m_attackerIndices.end();
        anim.UpdateNodeColor(n, isAtk?255:0, isAtk?0:255, isAtk?0:127);
        anim.UpdateNodeDescription(n, isAtk?"Attacker":"IoT");
        anim.UpdateNodeSize(n, isAtk?1.5:1.0, isAtk?1.5:1.0);
    }

    for (uint32_t i=0;i<m_baseStations.GetN();++i) {
        anim.UpdateNodeColor(m_baseStations.Get(i), 0,191,255);
        anim.UpdateNodeSize(m_baseStations.Get(i),4.0,4.0);
        std::stringstream ss; ss<<"Gateway "<<i+1;
        anim.UpdateNodeDescription(m_baseStations.Get(i), ss.str());
    }

    anim.UpdateNodeColor(m_serverNode.Get(0), 128,0,128);
    anim.UpdateNodeSize(m_serverNode.Get(0),6.0,6.0);
    anim.UpdateNodeDescription(m_serverNode.Get(0),"SERVER");

    SetupFlowMonitor();
    ScheduleRealtimeStats();
    SetupMitigation();

    std::cout << "\n🚀 STARTING SIMULATION..." << std::endl;
    Simulator::Stop(Seconds(m_simTime));
    Simulator::Run();
    m_anim = nullptr;
    Simulator::Destroy();
    std::cout << "✅ FINISHED." << std::endl;
}

int main(int argc, char* argv[])
{
    uint32_t n=30,a=10; double t=60.0;
    CommandLine cmd;
    cmd.AddValue("nodes","n",n);
    cmd.AddValue("attackers","a",a);
    cmd.AddValue("time","t",t);
    cmd.Parse(argc,argv);
    DDoSSimulator sim(n,a,t);
    sim.Run();
    return 0;
}