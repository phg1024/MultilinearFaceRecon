#include "MRSetting.h"
#include "Utils/stringutils.h"
#include "extra/tinyxml2/tinyxml2.h"

MRSetting::MRSetting(void)
{
}


MRSetting::~MRSetting(void)
{
}

bool MRSetting::load( const string& filename )
{
	typedef tinyxml2::XMLDocument	xml_doc_t;
	typedef tinyxml2::XMLElement	xml_elem_t;
	xml_doc_t doc;
	doc.LoadFile(filename.c_str());

	if( doc.ErrorID() ) {
		cerr << "Failed to load setting file " << filename << endl;
		return false;
	}

	xml_elem_t* titleElement = doc.FirstChildElement( "MRSettings" );

	xml_elem_t* paramElement = titleElement->FirstChildElement("param");
	while( paramElement != nullptr ) {
		string paramName = paramElement->Attribute("name");
		string paramType = paramElement->Attribute("type");
		string paramVal = paramElement->GetText();

		cout << "name = " << paramName << "\t" << "type = " << paramType << "\t" << "value = " << paramVal << endl;

		param_t::ParamType t = param_t::string2type(paramType);
		if( t == param_t::UNKNOWN ) {
			cerr << "Unknown parameter type! Please check the setting file " << filename << "!" << endl;
			return false;
		}
		else {
			param_t p;
			p.parseValue(t, paramVal);
			params.insert(make_pair(paramName, p));
		}

		paramElement = paramElement->NextSiblingElement("param");
	}

	return true;
}

bool MRSetting::save( const string& filename )
{
	// not implemented yet
	return false;
}

void MRSetting::print()
{
	for_each(params.begin(), params.end(), [](pair<string, param_t> x) {
		cout << x.first << '\t' << x.second.getValue() << endl;
	});
}
